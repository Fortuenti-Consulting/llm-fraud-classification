#!/usr/bin/env bash
# Proves verify.sh catches real bugs: re-injects bugs found in this project into a temp copy
# of the repo; every one must make verify.sh fail.
#   bash tests/mutants.sh
set -u
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
rsync -a --exclude .verify --exclude node_modules --exclude .venv --exclude venv --exclude __pycache__ ./ "$tmp/src/"
for d in node_modules .venv venv; do [ -e "$d" ] && ln -s "$PWD/$d" "$tmp/src/$d"; done  # deps shared, not copied
check() { (cd "$1" && VERIFY_SKIP_MUTANTS=1 VERIFY_ONLY="${2:-}" ./verify.sh >/dev/null 2>&1); }  # $2: only these ids

# A failing baseline would make every mutant look KILLED.
check "$tmp/src" || { echo "BASELINE FAILS: fix the suite before trusting mutants"; exit 2; }

survived=0
mutate() { # name, file, perl substitution (perl -0pi; escape @ and $ that are literal), [ids to run, e.g. V2]
  rm -rf "$tmp/m"; cp -R "$tmp/src" "$tmp/m"
  perl -0pi -e "$3" "$tmp/m/$2"
  if cmp -s "$tmp/src/$2" "$tmp/m/$2"; then echo "NOAPPLY  $1"; survived=$((survived+1)); return; fi
  if check "$tmp/m" "${4:-}"; then echo "SURVIVED $1"; survived=$((survived+1)); else echo "KILLED   $1"; fi
}

# Bugs found while preparing the public release (2026-10-10), re-injected one at a time.
mutate "rule engine reads field names the dataset lacks" card_fraud_analyzer.py 's/raw_ml\["velocity_pan_count_1m_ip"\]/raw_ml.get("card_velocity_60s", 1)/' V2
mutate "rule engine carding rule ignores the decline code" card_fraud_analyzer.py 's/ and features\["cvv_match"\] and features\["declined_code"\]/ and features["cvv_match"]/' V2
mutate "confusion matrix looks for one-shot logs" build_confusion_matrix.py 's/llm_eval_log_zero-shot_openai/llm_eval_log_one-shot_openai/' V3
mutate "missing log skipped instead of failing" build_confusion_matrix.py 's/sys\.exit\(f"Could not find[^\n]*\n/print("skipping")\n        continue\n/' V3
mutate "out-of-list CoT label crashes the matrix" build_confusion_matrix.py 's/confusion_matrix\.get\(cat_b, \{\}\)\.get\(cat_a, 0\)/confusion_matrix[cat_b].get(cat_a, 0)/' V3
mutate "Anthropic call left at default temperature" classify_llm_few-shot.py 's/ *extra_body=\{"temperature": 0\.0\},[^\n]*\n//' V5
mutate "a run scored on a different sample" llm_eval_log_zero-shot_openai.csv 's/tx_9bf5f27eda/tx_3fc1372a4e/' V6
mutate "report cites a transaction not in the logs" Engineer_Report.md 's/tx_9bf5f27eda/tx_ae318ef002/' V8
mutate "README results table out of date" README.md 's/90% · 531 ms/91% · 531 ms/' V8
mutate "shifted BIN testing back inside the rule's velocity range" dataset_generator.py 's/randint\(6, 10\)/randint(11, 20)/' V10
mutate "narrative calls any email domain disposable" dataset_generator.py 's/if raw\["email_domain"\] in DISPOSABLE_EMAIL_DOMAINS/if True/' V10
mutate "default mode consumes an extra random number" dataset_generator.py 's/if shifted and random\.random\(\) < SHIFT_RATE:/if random.random() < SHIFT_RATE and shifted:/' V11
mutate "README shifted category table out of date" README.md 's/\| 0 \| 0 \| 0 \| 1 \| 6 \|/| 0 | 0 | 0 | 1 | 7 |/' V8
mutate "shifted seed changed after publishing" dataset_generator.py 's/SHIFTED_SEED = 7/SHIFTED_SEED = 8/' V10
mutate "playground saved with stale outputs" playground.ipynb 's/88 of    88 labeled correctly/87 of    88 labeled correctly/' V12
mutate "playground live API call switched on by default" playground.ipynb 's/RUN_LIVE_CALL = False/RUN_LIVE_CALL = True/' V12
mutate "grid cell reads the wrong engine's log" cost-accuracy_analysis.py 's/llm_eval_log_\{strat\}_\{eng\}\.csv/llm_eval_log_{strat}_openai.csv/' V4

echo "survivors: $survived"
exit $(( survived > 0 ))
