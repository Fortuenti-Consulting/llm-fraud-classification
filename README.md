# LLM fraud classification: zero-shot vs. few-shot vs. chain-of-thought

Can a small LLM sort card transactions into fraud types fast and accurately enough to block them in real time? This repo is the full experiment behind that question: a synthetic dataset, three prompting strategies run on two models, a rule-based baseline, and the analysis scripts that produce every number below.

Short answer: few-shot prompting gets close to the best possible accuracy on this data, but every LLM setup is roughly 3-32x too slow for a sub-200 ms blocking path, and chain-of-thought made things slower, more expensive and less accurate.

Version: see [`VERSION`](VERSION).

## Results

Same 50 transactions for every run (10 per category, IDs in [`eval_sample_ids.txt`](eval_sample_ids.txt)), temperature 0 on both models. Cost is the total for the 50 calls at list prices (GPT-4o-mini $0.15/$0.60, Claude Haiku 4.5 $1/$5 per million input/output tokens).

| Strategy | GPT-4o-mini | Claude Haiku 4.5 |
|---|---|---|
| Zero-shot | 74% · 595 ms · $0.0025 | 74% · 640 ms · $0.0206 |
| Few-shot (2 examples) | 90% · 531 ms · $0.0040 | **92%** · 664 ms · $0.0321 |
| Chain-of-thought | 56% · 2,045 ms · $0.0092 | 74% · 6,458 ms · $0.1649 |
| Rule engine (baseline) | 47/50 = 94% · <0.01 ms · $0 | |

Accuracy · mean latency per transaction · total cost. Source: [`performance_grid_analysis.json`](performance_grid_analysis.json), [`confusion_matrix_analysis.json`](confusion_matrix_analysis.json), [`rule_engine_eval_log.csv`](rule_engine_eval_log.csv). On all 5,000 transactions the rule engine scores 99.5%.

What the errors show:

- **Zero-shot's main failure is Carding vs. BIN testing** (8-9 of 13 errors per model). Both are automated card attacks; the difference is in the verification signal (CVV mismatch = guessing numbers, CVV match plus issuer decline = stolen card lists). Two examples that spell out that difference remove all of them: few-shot has zero Carding/BIN testing errors on both models.
- **Three transactions are missed by every method, including the rule engine.** Two stolen-card records and one "sleeper" synthetic identity carry no fraud signal in their fields. That makes 47/50 (94%) the practical ceiling for this sample, and few-shot Claude Haiku 4.5 is one transaction below it.
- **Chain-of-thought hurt.** The scratchpads show GPT-4o-mini using "Carding" as an umbrella term for any card attack and fixating on new accounts with disposable emails, and Claude Haiku 4.5 reading "rotated through many cards" as BIN testing. Full breakdown with transaction IDs: [`Engineer_Report.md`](Engineer_Report.md). Executive summary: [`VP_Report.md`](VP_Report.md).

## Caveats

Read these before quoting the numbers.

1. **Synthetic data.** All 5,000 transactions come from [`dataset_generator.py`](dataset_generator.py). No real cardholder, merchant or network data is involved. IPs, ZIP codes and user agents are random values from [Faker](https://faker.readthedocs.io/); BINs are random or well-known test-card prefixes.
2. **n = 50.** One transaction is 2 percentage points. Treat differences under ~10 points as noise.
3. **The rule engine is partly circular.** Its rules were written by the same person who wrote the generator, so it knows each category's signature. Its 94-99.5% shows that structured signals beat an LLM when they are this clean, not that rules would do this well on real traffic.
4. **The chain-of-thought prompt is not a pure "add reasoning" test.** It lists the category names but not the definitions or examples the other two prompts have. Part of its drop comes from that. A few-shot + CoT prompt would isolate the effect of reasoning.
5. **Narratives lose information by design.** The LLMs read short text narratives built from the raw fields ([`enrich_for_llm`](dataset_generator.py)), not the fields themselves. For accounts 14 days or older, for example, the narrative drops the synthetic identity score.
6. **Latency was measured from a single machine** over the public APIs and includes network time.
7. **CoT output limits differ by model:** `max_tokens` is 300 for GPT-4o-mini and 800 for Claude Haiku 4.5. No response was cut off (every CoT output has its closing `</classification>` tag), but Haiku writes scratchpads about twice as long (2,350 vs. 1,050 characters on average), which adds to its CoT cost and latency on top of its higher token price.

## How it works

```
dataset_generator.py ──> synthetic_fraud_dataset.json / .csv   (5,000 transactions, 92% legitimate)
                               │
       ┌───────────────────────┼─────────────────────────────┐
       │                       │                             │
classify_llm_zero-shot.py  classify_llm_few-shot.py  classify_llm_chain-of-thought.py
       │      (each run on OpenAI and Anthropic, same 50 IDs)  │
       └───────────────> llm_eval_log_<strategy>_<engine>.csv <┘
                               │
       ┌───────────────────────┼─────────────────────────┐
build_confusion_matrix.py  cost-accuracy_analysis.py    card_fraud_analyzer.py (rule baseline)
       │                       │                             │
confusion_matrix_analysis.json  performance_grid_analysis.json   rule_engine_eval_log.csv
                               │
                     generate_reports.py ──> VP_Report.md, Engineer_Report.md
```

1. **Generate.** Five categories with a realistic imbalance (92% legitimate, 2% each for BIN testing, carding, stolen card, synthetic identity). 12% of records get 2-3 fields changed to create hard edge cases (a legitimate gift shipped to another ZIP, a synthetic account aged for months). Each record is then turned into a six-part narrative for the LLM.
2. **Classify.** Each script sends one prompt per transaction and logs the prediction, latency, tokens and cost.
3. **Analyze.** Confusion matrices per run, the accuracy/latency/cost grid, and a rule-based baseline over the same fields.
4. **Report.** Both reports are generated from the logs, so their numbers always match the data.

## Run it

Python 3.10+.

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Put your OpenAI and Anthropic keys in `.env`. Run everything from the repo root. The committed dataset and logs let you skip straight to step 3.

```bash
python3 dataset_generator.py
```

Regenerates the dataset. Seeds fix the sampled values, but transaction IDs and timestamps change on every run, so `eval_sample_ids.txt` will no longer match. Skip this to reproduce the published results.

```bash
ENGINE=OPENAI python3 classify_llm_few-shot.py
```

Use `ENGINE=ANTHROPIC` for Claude, and the `zero-shot` / `chain-of-thought` scripts for the other strategies. A full set of 6 runs costs about $0.25.

```bash
python3 build_confusion_matrix.py && python3 cost-accuracy_analysis.py && python3 card_fraud_analyzer.py && python3 generate_reports.py
```

## Files

| Path | What it is |
|---|---|
| `dataset_generator.py` | Builds the synthetic dataset and the LLM narratives |
| `synthetic_fraud_dataset.json` / `.csv` | The 5,000 transactions (nested for the LLM scripts, flat for tabular ML) |
| `eval_sample_ids.txt` | The 50 transaction IDs every LLM run uses |
| `classify_llm_*.py` | One script per prompting strategy |
| `llm_eval_log_*.csv` | Per-transaction results of the 6 runs, including the full prompt or CoT output |
| `build_confusion_matrix.py` | Confusion matrices, error asymmetries and never-confused pairs |
| `cost-accuracy_analysis.py` | Accuracy / latency / cost grid |
| `card_fraud_analyzer.py` | Deterministic rule engine baseline |
| `generate_reports.py` | Writes `VP_Report.md` and `Engineer_Report.md` from the data |
| `archive/` | First few-shot runs (Sep 2026). OpenAI: an earlier run of the published few-shot setup on the same 50 IDs (the shared sample was taken from it), with identical predictions and slower latencies. Anthropic: a different random sample at the default temperature. Not used in the results |
| `tests/`, `verify.sh` | Checks that the analysis code and reports are correct (below) |

## Verification

`./verify.sh` runs offline in a few seconds. It checks that no secrets are in the repo, tests the rule engine, matrix and grid code against hand-computed fixtures, confirms all six logs use the same 50 transactions, checks that the reports match the logs, and re-injects past bugs to confirm the checks catch them. The checklist is in [`CLAUDE.md`](CLAUDE.md).

## License

[MIT](LICENSE)
