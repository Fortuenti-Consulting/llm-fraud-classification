"""
Write the two reports from the results.

VP_Report.md is a one-page summary for a decision maker. Engineer_Report.md
lists every chain-of-thought mistake with the payments involved and the reason
behind it. All numbers are read from the result files, so the reports stay
correct when results change; only the root-cause sentences in ROOT_CAUSES are
written by hand, from reading the models' scratchpads.

How to run (from the project folder, after cost-accuracy_analysis.py):
  python3 generate_reports.py
"""
import csv
import json
from collections import defaultdict

# 1. Load the analysis data
with open("performance_grid_analysis.json", "r") as f:
    perf_data = json.load(f)

ENGINE_NAMES = {"openai": "GPT-4o-mini", "anthropic": "Claude Haiku 4.5"}
STRATEGY_NAMES = {"zero-shot": "zero-shot", "few-shot": "few-shot", "cot": "chain-of-thought"}

cells = [(s, e, perf_data[s][e]) for s in perf_data for e in perf_data[s]]
fastest_s, fastest_e, fastest = min(cells, key=lambda c: c[2]["latency"])
best_fs_e = max(perf_data["few-shot"], key=lambda e: perf_data["few-shot"][e]["accuracy"])
best_fs = perf_data["few-shot"][best_fs_e]
cot_slowest = max(perf_data["cot"].values(), key=lambda c: c["latency"])

# Few-shot + CoT on the same 50 transactions (classify_llm_few-shot-cot.py)
fs_cot_acc = {}
for engine in ["openai", "anthropic"]:
    with open(f"llm_eval_log_few-shot-cot_{engine}.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fs_cot_acc[engine] = 100 * sum(r["true_label"] == r["predicted_label"] for r in rows) / len(rows)

# Rule engine on the same 50, and the shifted stress test (50 records built to slip past the rules)
def sample_hits(log, ids_file):
    with open(ids_file) as f:
        ids = set(f.read().split())
    with open(log, encoding="utf-8") as f:
        return sum(r["is_correct"] == "True" for r in csv.DictReader(f) if r["transaction_id"] in ids)

rule_hits = sample_hits("rule_engine_eval_log.csv", "eval_sample_ids.txt")
shifted_rule_hits = sample_hits("shifted_rule_engine_eval_log.csv", "shifted_sample_ids.txt")
shifted_acc = {}
for strategy in ["few-shot", "few-shot-cot"]:
    for engine in ["openai", "anthropic"]:
        with open(f"llm_eval_log_{strategy}_{engine}_shifted.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        shifted_acc[(strategy, engine)] = 100 * sum(r["true_label"] == r["predicted_label"] for r in rows) / len(rows)
best_shifted = max(shifted_acc, key=shifted_acc.get)
best_shifted_style = "reasoning step by step" if best_shifted[0] == "few-shot-cot" else "few-shot"

# 2. VP report
vp_report = f"""# LLM fraud classification: summary for decision makers

## What was tested
Two language models, GPT-4o-mini and Claude Haiku 4.5, labeled 50 made-up card payments (10 per category, the same 50 for every setup) using three prompt styles. Two questions: can they sort a short text description of a payment into 5 categories, and are they fast enough to block a payment while it is being made? All data is synthetic; read the caveats in the README before applying any of these numbers to real traffic.

## Findings
1. **Too slow to block payments in real time.** The fastest setup ({ENGINE_NAMES[fastest_e]}, {STRATEGY_NAMES[fastest_s]}) averaged {fastest['latency']:.0f}ms per transaction. A decision to block a card payment usually has to come back in under 200ms.
2. **Two examples closed most of the gap.** Adding 2 worked examples to the prompt (few-shot) raised accuracy from {perf_data['zero-shot']['openai']['accuracy']:.0f}% to {perf_data['few-shot']['openai']['accuracy']:.0f}% on GPT-4o-mini and from {perf_data['zero-shot']['anthropic']['accuracy']:.0f}% to {perf_data['few-shot']['anthropic']['accuracy']:.0f}% on Claude Haiku 4.5. Best result: {best_fs['accuracy']:.0f}% ({ENGINE_NAMES[best_fs_e]}). A short list of hand-written rules got {rule_hits} of the same 50 right in well under a millisecond.
3. **Asking the model to reason step by step did not pay off on this sample.** Latency rose to as much as {cot_slowest['latency']:.0f}ms, cost multiplied, and accuracy fell below few-shot ({perf_data['cot']['openai']['accuracy']:.0f}% on GPT-4o-mini, {perf_data['cot']['anthropic']['accuracy']:.0f}% on Claude Haiku 4.5). The written reasoning shows the models dwelling on secondary signals. One caveat: that prompt listed only the category names, without the definitions or examples. With them added (few-shot + chain-of-thought), accuracy returns to {fs_cot_acc['openai']:.0f}% on GPT-4o-mini and {fs_cot_acc['anthropic']:.0f}% on Claude Haiku 4.5, so most of the drop came from the missing definitions. The reasoning step itself added time and cost without improving accuracy here.
4. **When the fraud changed, the models held up better than the rules.** On 50 payments built to slip past the rules, the rules got {shifted_rule_hits} right, as designed. The four few-shot setups scored {min(shifted_acc.values()):.0f}-{max(shifted_acc.values()):.0f}%, best with {ENGINE_NAMES[best_shifted[1]]} {best_shifted_style} ({shifted_acc[best_shifted]:.0f}%). Rules are fast and cheap but only catch what someone already thought of.

## What to do next
1. Keep LLMs out of the real-time blocking path.
2. Turn the signals this pilot surfaced (card checks, velocity, account age, shipping address) into features for a conventional model such as XGBoost or a random forest, trained and tested on real labeled transactions. Such a model can answer in under 200ms.
3. Before trusting any model or rule set, test it on cases built to break it, the way the shifted set does here.
4. Possible side use: run the few-shot prompt after the fact, as a "shadow" pipeline that writes a plain-language explanation for the fraud review team once a transaction has been blocked.
"""

# 3. Engineer report
# Root causes come from reading the <scratchpad> text of the misclassified rows.
# Keyed by (engine, true label, predicted label); pathways without a note are listed without one.
ROOT_CAUSES = {
    ("anthropic", "Carding", "BIN_Testing"):
        "The scratchpads treat \"rotated through N different cards\" as the defining sign of BIN testing. They mention the "
        "CVV match and issuer decline (Code 51) but read it as part of the testing, not as proof the attacker already had "
        "the card details. Several scratchpads even note that Carding could apply, then pick BIN_Testing as \"more specific\". "
        "The CoT prompt lists only the category names, not their definitions, so nothing ties CVV match to Carding.",
    ("anthropic", "Legitimate", "Stolen_Card"):
        "A noisy legitimate order (brand-new guest checkout, proxy IP, shipping ZIP different from billing). The scratchpad "
        "reads the passed CVV/AVS checks as proof the fraudster holds full card details. From the narrative alone this is "
        "hard to tell apart from a stolen card; a gift shipped to another ZIP looks the same.",
    ("anthropic", "Stolen_Card", "Carding"):
        "Established account, clean authorization, drop-house shipping. The scratchpad rules out Stolen_Card because "
        "verification passed, then calls it Carding, meaning a stolen card buying goods for a drop-house. Without definitions in the "
        "prompt, the model's own meaning of Carding wins.",
    ("anthropic", "Stolen_Card", "Legitimate"):
        "Noisy stolen-card records: billing and shipping ZIPs match, no proxy, normal velocity, established account. "
        "The narrative carries no fraud signal, and the scratchpads describe it correctly. No prompt can recover these; "
        "the signal is lost when the raw fields become narrative.",
    ("anthropic", "Synthetic_Identity", "Legitimate"):
        "A \"sleeper\" synthetic account (245 days old, gmail.com, synthetic score 0.508). For accounts 14 days or older the "
        "identity narrative only says \"Established user account\" and drops the score, so the model never sees it.",
    ("openai", "BIN_Testing", "Carding"):
        "The scratchpads list every BIN-testing signal correctly (N7 CVV mismatch, 20-58 card numbers in 60 seconds on one "
        "BIN, pre-auth or micro amount) and then call it Carding. The model uses \"carding\" as the umbrella term for any "
        "card-testing attack; the CoT prompt gives no definitions to separate the two.",
    ("openai", "Carding", "Synthetic_Identity"):
        "Identity fixation: the scratchpads focus on the 1-2 day old account and the disposable email domain and conclude "
        "Synthetic_Identity, giving little weight to the 9-20 card rotation and the CVV-matched issuer decline that define Carding.",
    ("openai", "Legitimate", "Stolen_Card"):
        "Same noisy legitimate order as in the Claude Haiku 4.5 section. The scratchpad sees clean authorization and normal velocity but lets the "
        "billing/shipping mismatch decide.",
    ("openai", "Stolen_Card", "Carding"):
        "Established accounts with clean authorizations and drop-house shipping. The scratchpads spot the drop-house "
        "pattern and label it Carding, again using the term for any purchase with stolen card details.",
    ("openai", "Stolen_Card", "Legitimate"):
        "Same records as the Claude Haiku 4.5 errors: the narrative has no fraud signal, and the scratchpads read it correctly.",
    ("openai", "Synthetic_Identity", "Legitimate"):
        "Same sleeper account as the Claude Haiku 4.5 error: the synthetic identity score is not in the narrative.",
}

errors = {}
for engine in ["anthropic", "openai"]:
    pathways = defaultdict(list)
    with open(f"llm_eval_log_cot_{engine}.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["true_label"] != row["predicted_label"]:
                pathways[(row["true_label"], row["predicted_label"])].append(row["transaction_id"])
    errors[engine] = sorted(pathways.items(), key=lambda p: (-len(p[1]), p[0]))

total_errors = sum(len(ids) for engine in errors for _, ids in errors[engine])
engineer_report = f"""# Chain-of-thought errors, one by one
All {total_errors} wrong answers from the two chain-of-thought runs on the 50-transaction sample (the prompt without category definitions), with the reason for each, read from the models' scratchpads in `llm_eval_log_cot_openai.csv` and `llm_eval_log_cot_anthropic.csv`. The few-shot + chain-of-thought runs added in v1.1.0 are not broken down here.
"""
for engine in ["anthropic", "openai"]:
    n = sum(len(ids) for _, ids in errors[engine])
    engineer_report += f"\n## {ENGINE_NAMES[engine]} ({n} errors)\n"
    for (true_label, pred_label), ids in errors[engine]:
        engineer_report += f"\n### {true_label} predicted as {pred_label} ({len(ids)})\n"
        engineer_report += "* **Transactions:** " + ", ".join(f"`{i}`" for i in ids) + "\n"
        note = ROOT_CAUSES.get((engine, true_label, pred_label))
        if note:
            engineer_report += f"* **Root cause:** {note}\n"

engineer_report += """
## What this means for the next model
When two signals pointed different ways (high card velocity but a brand-new account, say), the models did not weigh them the same way twice. The raw signals (account age, velocity count, CVV match, ZIP mismatch) belong in a table as features, so a model such as XGBoost can learn how they combine.
"""

# 4. Write reports to disk
with open("VP_Report.md", "w", encoding="utf-8") as f:
    f.write(vp_report)

with open("Engineer_Report.md", "w", encoding="utf-8") as f:
    f.write(engineer_report)

print("Generated VP_Report.md and Engineer_Report.md")
