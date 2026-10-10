# LLM fraud classification study

Python scripts, run from the repo root. Pipeline and file map: `README.md`. No API calls happen in `./verify.sh`.

## Verify

Tier T1 (analytics). Run `./verify.sh`; done = exit 0 and every `[auto]` item prints PASS. Never weaken a test, skip one, loosen a tolerance, or generate expected values from the app's own output.

- **V1** [auto] Nothing a push would publish contains `.env`, API-key patterns, private keys, home-directory paths or personal emails; `.env` is gitignored.
- **V2** [auto] Rule engine reads the dataset's real fields: hand-built records hit every rule and its boundary; on the full dataset it predicts all 5 categories.
- **V3** [auto] `build_confusion_matrix.py` on hand-made logs gives hand-counted matrices, accuracy, asymmetry, winner and zero-error cells; handles out-of-list CoT labels; a missing log fails the run.
- **V4** [auto] `cost-accuracy_analysis.py` on hand-made logs gives hand-computed accuracy, mean latency and total cost per cell.
- **V5** [auto] CoT label extraction handles tags, case and whitespace and returns `ERROR_PARSING` without tags; every classify script's source uses `eval_sample_ids.txt` and sets temperature 0 on both engines (static check; no API calls).
- **V6** [auto] All 6 committed eval logs cover exactly the 50 IDs in `eval_sample_ids.txt` (10 per category) with the dataset's true labels.
- **V7** [auto] `enrich_for_llm` turns hand-built raw records into the expected narrative for each branch.
- **V8** [auto] Published numbers match the data: README results table, rule-engine figures and VP numbers equal values recomputed from the logs; every transaction cited in `Engineer_Report.md` is a CoT misclassification in the named log and none is missing; regenerating the reports, matrices and rule log changes nothing.
- **VM** [auto] `tests/mutants.sh`: bugs found before the public release are re-injected; every one must be caught.
- **V9** [manual: Ricardo, before each article update] Read the reports and README results against `cost-accuracy_analysis.py` output; root-cause text in `generate_reports.py` still matches the CoT scratchpads.
