import os
from dotenv import load_dotenv
import json
import time
import pandas as pd
import httpx2
from openai import OpenAI
from anthropic import Anthropic

# Load API keys from the .env file next to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path, override=True)

# 1. Engine configuration
ENGINE_CHOICE = os.getenv("ENGINE", "OPENAI").upper()  # ENGINE=ANTHROPIC python3 <script>

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    raise ValueError(f"Missing API keys. Looked for .env at: {env_path}")

# DATASET=shifted runs on the shifted stress-test set instead of the published one
SHIFTED = os.getenv("DATASET", "") == "shifted"
DATASET_FILE = "shifted_fraud_dataset.json" if SHIFTED else "synthetic_fraud_dataset.json"
SAMPLE_FILE = "shifted_sample_ids.txt" if SHIFTED else "eval_sample_ids.txt"

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# USD per million tokens
PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00}
}

# trust_env=False ignores any system proxy settings
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx2.Client(trust_env=False)
)

anthropic_client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url="https://api.anthropic.com",
    http_client=httpx2.Client(trust_env=False)
)

# 2. Prompt
# Same prompt as zero-shot plus two worked examples for the pair the models
# confuse most: BIN testing vs. carding.
def build_few_shot_prompt(narrative_dict: dict) -> str:
    prompt = f"""You are an expert financial fraud investigator analyzing a real-time transaction.
Your task is to classify the transaction into exactly ONE of the following 5 categories based on the narrative provided.

CATEGORIES & DEFINITIONS:
1. Legitimate: A standard, valid transaction with no malicious intent.
2. BIN_Testing: High-speed, automated micro-transactions guessing card numbers.
3. Carding: Scripted low-value transactions validating stolen credentials.
4. Stolen_Card: High-value unauthorized purchases using stolen card details.
5. Synthetic_Identity: Transactions tied to newly fabricated identities.

--- EXAMPLES FOR CLASSIFYING AUTOMATED ATTACKS ---
Because BIN Testing and Carding look similar, use these exact rules:

EXAMPLE 1
Narrative Clues: "Authorization failed due to a CVV Mismatch (Code N7)" AND "triggering massive CVV failures on BIN X"
Correct Classification: BIN_Testing
Reasoning: BIN Testing brute-forces the 3-digit CVV, resulting in explicit CVV Mismatches. It targets a single BIN prefix.

EXAMPLE 2
Narrative Clues: "CVV matched, but the transaction was declined... (Code 51 - Insufficient Funds / Code 05 - Do Not Honor)" AND "rotated through X different cards"
Correct Classification: Carding
Reasoning: Carders possess stolen lists containing the real CVVs. The CVV matches, but the issuer declines it because the stolen card is empty or blocked. They rotate through many different cards, not a single BIN.
--------------------------------------------------

TRANSACTION NARRATIVE:
{json.dumps(narrative_dict, indent=2)}

INSTRUCTIONS:
Evaluate the narrative and return ONLY the exact category name from the list above. Do not include explanations, punctuation, or extra text.
"""
    return prompt

# 3. LLM call, cost and latency
def classify_with_llm(prompt: str, engine: str):
    start_time = time.perf_counter()

    if engine == "OPENAI":
        active_model = OPENAI_MODEL
        response = openai_client.chat.completions.create(
            model=active_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        predicted_label = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

    elif engine == "ANTHROPIC":
        active_model = ANTHROPIC_MODEL
        response = anthropic_client.messages.create(
            model=active_model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"temperature": 0.0},  # SDK 1.x no longer takes temperature as an argument
            max_tokens=10
        )
        predicted_label = response.content[0].text.strip()
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

    else:
        raise ValueError(f"Unsupported engine: {engine}")

    latency_ms = (time.perf_counter() - start_time) * 1000

    cost = (
        (prompt_tokens / 1_000_000) * PRICING[active_model]["input"] +
        (completion_tokens / 1_000_000) * PRICING[active_model]["output"]
    )

    return predicted_label, prompt_tokens, completion_tokens, cost, latency_ms

# 4. Main
if __name__ == "__main__":
    print(f"Loading {DATASET_FILE}...")
    with open(DATASET_FILE, "r") as f:
        dataset = json.load(f)

    # Every strategy is scored on the same 50 transactions (10 per category)
    with open(SAMPLE_FILE, "r") as f:
        sample_ids = f.read().split()
    records_by_id = {r["raw_ml_features"]["transaction_id"]: r for r in dataset}
    sampled_records = [records_by_id[tx_id] for tx_id in sample_ids]

    print(f"Loaded {len(sampled_records)} records. Starting classification via {ENGINE_CHOICE}...")

    results_log = []
    correct_predictions = 0
    total_cost = 0.0

    for i, record in enumerate(sampled_records, 1):
        true_label = record["raw_ml_features"]["fraud_category"]
        narrative = record["llm_reasoning_prompt"]
        tx_id = record["raw_ml_features"]["transaction_id"]

        prompt = build_few_shot_prompt(narrative)

        try:
            predicted_label, p_tok, c_tok, call_cost, latency_ms = classify_with_llm(prompt, ENGINE_CHOICE)

            total_cost += call_cost
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_predictions += 1

            print(f"[{i}/{len(sampled_records)}] {tx_id} | True: {true_label} | Pred: {predicted_label} | Latency: {latency_ms:.0f}ms | Correct: {is_correct}")

            results_log.append({
                "transaction_id": tx_id,
                "engine_used": ENGINE_CHOICE,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "cost_usd": call_cost,
                "prompt_sent": prompt
            })

        except Exception as e:
            print(f"Error classifying {tx_id} with {ENGINE_CHOICE}: {e}")

    # 5. Summary
    # Failed API calls count as incorrect: the denominator is the full sample
    df_results = pd.DataFrame(results_log)
    accuracy = (correct_predictions / len(sampled_records)) * 100
    avg_latency = df_results["latency_ms"].mean()
    p95_latency = df_results["latency_ms"].quantile(0.95)
    max_latency = df_results["latency_ms"].max()

    print("\n" + "="*45)
    print(f"EVALUATION COMPLETE: {ENGINE_CHOICE}")
    print("="*45)
    print(f"Total Evaluated: {len(sampled_records)}")
    print(f"Accuracy:        {accuracy:.1f}% ({correct_predictions}/{len(sampled_records)})")
    print(f"Total Cost:      ${total_cost:.6f}")
    print(f"Avg Latency:     {avg_latency:.0f} ms")
    print(f"p95 Latency:     {p95_latency:.0f} ms")
    print(f"Max Latency:     {max_latency:.0f} ms")
    print("="*45)

    output_file = f"llm_eval_log_few-shot_{ENGINE_CHOICE.lower()}{'_shifted' if SHIFTED else ''}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed log saved to {output_file}")
