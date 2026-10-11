#!/usr/bin/env bash
# Definition of done for this repo. Exit 0 = every [auto] item in CLAUDE.md "## Verify" passed.
# No API calls: checks run against the committed logs and hand-made fixtures.
set -u
cd "$(dirname "$0")"
rc=0
python3 -c "import pandas, openai, anthropic, httpx2, faker, dotenv, nbclient, nbformat" 2>/dev/null \
  || { echo "VERIFY_ENV_UNAVAILABLE python deps missing (pip install -r requirements.txt)"; exit 1; }

check() { # id, command...  — prints PASS/FAIL id; output shown only on failure
  local id=$1; shift
  # mutants.sh narrows a run (VERIFY_ONLY + VERIFY_SKIP_MUTANTS); a stray VERIFY_ONLY alone never skips checks
  if [ -n "${VERIFY_ONLY:-}" ] && [ -n "${VERIFY_SKIP_MUTANTS:-}" ] && [[ ",$VERIFY_ONLY," != *",$id,"* ]]; then return 0; fi
  if out=$("$@" 2>&1); then echo "PASS $id"; else echo "FAIL $id"; printf '%s\n' "$out" | tail -15; rc=1; fi
}

for id in V1 V2 V3 V4 V5 V6 V7 V8 V10 V11 V12; do
  check "$id" python3 tests/test_pipeline.py "$id"
done

# Mutation check last; mutants.sh calls verify.sh with VERIFY_SKIP_MUTANTS=1, so no recursion.
if [ -z "${VERIFY_SKIP_MUTANTS:-}" ]; then
  if bash tests/mutants.sh; then echo "PASS VM"; else echo "FAIL VM"; rc=1; fi
fi
exit $rc
