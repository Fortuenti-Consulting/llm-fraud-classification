import csv
import json
from collections import defaultdict

# 1. Load the analysis data
with open("performance_grid_analysis.json", "r") as f:
    perf_data = json.load(f)

ENGINE_NAMES = {"openai": "GPT-4o-mini", "anthropic": "Claude Haiku 4.5"}
STRATEGY_NAMES = {"zero-shot": "Zero-Shot", "few-shot": "Few-Shot", "cot": "Chain-of-Thought"}

cells = [(s, e, perf_data[s][e]) for s in perf_data for e in perf_data[s]]
fastest_s, fastest_e, fastest = min(cells, key=lambda c: c[2]["latency"])
best_fs_e = max(perf_data["few-shot"], key=lambda e: perf_data["few-shot"][e]["accuracy"])
best_fs = perf_data["few-shot"][best_fs_e]
cot_slowest = max(perf_data["cot"].values(), key=lambda c: c["latency"])

# 2. VP report
vp_report = f"""# Executive Brief: LLM Fraud Classification Pilot Study

## Overview & Scope
We ran an n=50 pilot (10 transactions per category, the same 50 for every configuration) to evaluate Large Language Models (LLMs) for real-time transaction fraud classification. The goal was to understand if LLMs could accurately classify transactions into 5 categories from narratives built on categorical and behavioral signals, and whether their latency supports synchronous blocking.

## Key Findings
1. **LLMs are too slow for synchronous blocking.** The fastest configuration ({ENGINE_NAMES[fastest_e]}, {STRATEGY_NAMES[fastest_s]}) averaged {fastest['latency']:.0f}ms per transaction. Our synchronous blocking layer requires a sub-200ms response time.
2. **Zero-Shot vs. Few-Shot:** LLMs need explicit examples to apply specialized fraud definitions. Adding 2 examples (Few-Shot) raised accuracy from {perf_data['zero-shot']['openai']['accuracy']:.0f}% to {perf_data['few-shot']['openai']['accuracy']:.0f}% on GPT-4o-mini and from {perf_data['zero-shot']['anthropic']['accuracy']:.0f}% to {perf_data['few-shot']['anthropic']['accuracy']:.0f}% on Claude Haiku 4.5. Peak accuracy: {best_fs['accuracy']:.0f}% ({ENGINE_NAMES[best_fs_e]}).
3. **Chain-of-Thought (CoT) degrades performance:** Asking the model to reason step by step before classifying raised average latency to as much as {cot_slowest['latency']:.0f}ms, multiplied cost, and *reduced* accuracy compared with Few-Shot ({perf_data['cot']['openai']['accuracy']:.0f}% on GPT-4o-mini, {perf_data['cot']['anthropic']['accuracy']:.0f}% on Claude Haiku 4.5). The scratchpads show the models overthinking secondary signals. Caveat: the CoT prompt listed only the category names, without the definitions or examples, so part of the drop comes from that and not from step-by-step reasoning alone.

## Recommendations & Next Steps
1. **Do not deploy LLMs in the synchronous path.**
2. **Proceed to Traditional ML:** Use the insights from this pilot to engineer features for a traditional Machine Learning model (e.g., XGBoost or Random Forest), which can meet the sub-200ms latency requirement.
3. **Expand Evaluation:** Scale testing from 50 samples to a larger dataset (10,000+) to train and validate the traditional ML layer.
4. **Potential Asynchronous Use Case:** The Few-Shot prompt can run as an asynchronous "shadow" pipeline that writes human-readable narratives for the fraud review team after the ML model blocks a transaction.
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
        "verification passed, then uses Carding for a stolen card used to buy goods shipped to a drop-house. Without definitions in the "
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
engineer_report = f"""# Engineering Failure Analysis: Chain-of-Thought Misclassifications
**Scope:** Root-cause analysis of all {total_errors} Chain-of-Thought misclassifications in the n=50 pilot, using the model scratchpads in `llm_eval_log_cot_openai.csv` and `llm_eval_log_cot_anthropic.csv`.
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
## Next Steps for ML Pipeline
The models' failure to weigh conflicting signals (e.g., velocity vs. identity) shows that free-text reasoning drifts. The raw signals (account age, velocity count, CVV match, ZIP mismatch) should become tabular features so a model such as XGBoost can learn how they interact.
"""

# 4. Write reports to disk
with open("VP_Report.md", "w", encoding="utf-8") as f:
    f.write(vp_report)

with open("Engineer_Report.md", "w", encoding="utf-8") as f:
    f.write(engineer_report)

print("Generated VP_Report.md and Engineer_Report.md")
