"""Checks behind ./verify.sh. Usage: python3 tests/test_pipeline.py V2

Expected values are worked out by hand from the inputs written here,
never copied from the scripts' own output.
"""
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CATEGORIES = ["Legitimate", "BIN_Testing", "Carding", "Stolen_Card", "Synthetic_Identity"]
STRATEGIES = ["zero-shot", "few-shot", "cot"]
ENGINES = ["openai", "anthropic"]


def git_files():
    """Files that a push would publish: tracked plus untracked-but-not-ignored."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if os.path.isfile(os.path.join(ROOT, f))]


def v1_no_secrets():
    files = git_files()
    assert files, "git sees no files"
    assert not [f for f in files if os.path.basename(f) == ".env"], ".env would be published"
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode
    assert ignored == 0, ".env is not gitignored"
    patterns = re.compile(
        r"sk-[A-Za-z0-9_-]{20,}|sk-" "ant-|AIza[0-9A-Za-z_-]{30}|ghp_[A-Za-z0-9]{20}|github_" "pat_"
        r"|BEGIN [A-Z ]*PRIVATE KEY|/Users/[a-z]|@gmail\.com")
    hits = []
    for f in files:
        with open(os.path.join(ROOT, f), "r", encoding="utf-8", errors="ignore") as fh:
            for n, line in enumerate(fh, 1):
                m = patterns.search(line)
                if m:
                    hits.append(f"{f}:{n}: {m.group(0)[:12]}...")
    assert not hits, "possible secrets or personal data:\n" + "\n".join(hits[:10])


def raw_record(**overrides):
    """A clean legitimate transaction; each test changes only the fields it needs."""
    raw = {
        "transaction_id": "tx_test", "fraud_category": "Legitimate", "amount": 42.0,
        "cvv_result": "Match", "auth_response_code": "00 - Approved",
        "velocity_pan_count_1m_ip": 1, "account_age_days": 400, "email_domain": "gmail.com",
        "identity_synthetic_score": 0.10, "billing_zip": "10001", "shipping_zip": "10001",
        "proxy_vpn_flag": False, "session_duration_ms": 30000, "transaction_type": "Auth-and-Capture",
        "currency": "USD", "bin": "411111", "field_input_method": "Manual Typing",
        "velocity_cvv_fail_count_5m_bin": 0,
    }
    raw.update(overrides)
    return {"raw_ml_features": raw}


def v2_rule_engine():
    import card_fraud_analyzer as c

    def label(**kw):
        return c.rule_based_fallback_classifier(c.extract_features(raw_record(**kw)))

    assert label() == "Legitimate"
    assert label(velocity_pan_count_1m_ip=11, cvv_result="Mismatch",
                 auth_response_code="N7 - CVV Mismatch") == "BIN_Testing"
    # velocity must be strictly above 10 for BIN testing; 10 with a mismatch falls through
    assert label(velocity_pan_count_1m_ip=10, cvv_result="Mismatch",
                 auth_response_code="N7 - CVV Mismatch") == "Legitimate"
    assert label(velocity_pan_count_1m_ip=6, auth_response_code="51 - Insufficient Funds") == "Carding"
    assert label(velocity_pan_count_1m_ip=6, auth_response_code="05 - Do Not Honor") == "Carding"
    # high velocity with a matched CVV but an approval is not carding
    assert label(velocity_pan_count_1m_ip=6) == "Legitimate"
    assert label(identity_synthetic_score=0.81) == "Synthetic_Identity"
    assert label(email_domain="temp-mail.org", account_age_days=13) == "Synthetic_Identity"
    assert label(email_domain="temp-mail.org", account_age_days=14) == "Legitimate"
    assert label(shipping_zip="94105", account_age_days=31) == "Stolen_Card"
    assert label(shipping_zip="94105", account_age_days=30) == "Legitimate"

    # On the real dataset the engine must use more than the default branch
    dataset = c.load_dataset(os.path.join(ROOT, "synthetic_fraud_dataset.json"))
    predicted = Counter(c.rule_based_fallback_classifier(c.extract_features(r)) for r in dataset)
    assert set(predicted) == set(CATEGORIES), f"predicted classes: {dict(predicted)}"


# Hand-made logs: 4 rows per file. Expected numbers below are counted by hand.
FIXTURE = {
    "zero-shot_openai":    [("Carding", "BIN_Testing"), ("Carding", "BIN_Testing"), ("BIN_Testing", "Carding"), ("Legitimate", "Legitimate")],
    "zero-shot_anthropic": [("Legitimate", "Legitimate"), ("Carding", "Carding"), ("Stolen_Card", "Legitimate"), ("Legitimate", "Legitimate")],
    "few-shot_openai":     [("Legitimate", "Legitimate"), ("Carding", "Carding"), ("BIN_Testing", "BIN_Testing"), ("Stolen_Card", "Stolen_Card")],
    "few-shot_anthropic":  [("Legitimate", "Legitimate"), ("Carding", "Carding"), ("BIN_Testing", "BIN_Testing"), ("Synthetic_Identity", "Legitimate")],
    "cot_openai":          [("Legitimate", "Drop_House"), ("Carding", "Carding"), ("BIN_Testing", "BIN_Testing"), ("Stolen_Card", "Stolen_Card")],
    "cot_anthropic":       [("Legitimate", "Legitimate"), ("Carding", "BIN_Testing"), ("BIN_Testing", "BIN_Testing"), ("Stolen_Card", "Stolen_Card")],
}
LATENCIES = [100.0, 200.0, 300.0, 600.0]  # mean 300, times the file's position in FIXTURE
COSTS = [0.001, 0.002, 0.003, 0.004]      # total 0.010, times the file's position in FIXTURE


def fixture_dir(script):
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(ROOT, script), tmp)
    for name, rows in FIXTURE.items():
        with open(os.path.join(tmp, f"llm_eval_log_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["transaction_id", "true_label", "predicted_label", "is_correct", "latency_ms", "cost_usd"])
            k = list(FIXTURE).index(name) + 1  # per-file scale, so a cell reading the wrong file is caught
            for i, (t, p) in enumerate(rows):
                w.writerow([f"tx_{i}", t, p, t == p, LATENCIES[i] * k, COSTS[i] * k])
    return tmp


def run(tmp, script):
    return subprocess.run([sys.executable, script], cwd=tmp, capture_output=True, text=True)


def v3_confusion_matrix():
    tmp = fixture_dir("build_confusion_matrix.py")
    r = run(tmp, "build_confusion_matrix.py")
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.load(open(os.path.join(tmp, "confusion_matrix_analysis.json")))
    per = data["per_file_analysis"]
    assert sorted(per) == sorted(f"llm_eval_log_{n}.csv" for n in FIXTURE), sorted(per)

    zs = per["llm_eval_log_zero-shot_openai.csv"]
    assert (zs["correct_predictions"], zs["accuracy_percent"]) == (1, 25.0)
    assert zs["raw_matrix"]["Carding"]["BIN_Testing"] == 2
    assert zs["raw_matrix"]["BIN_Testing"]["Carding"] == 1
    asym = zs["greatest_asymmetric_confusion"]
    assert (asym["true_label"], asym["predicted_label"], asym["difference"]) == ("Carding", "BIN_Testing", 1)

    # An out-of-list CoT label gets its own column instead of being dropped
    cot = per["llm_eval_log_cot_openai.csv"]
    assert cot["raw_matrix"]["Legitimate"]["Drop_House"] == 1 and cot["accuracy_percent"] == 75.0

    g = data["global_summary"]
    assert g["tightest_diagonal_winner"] == "llm_eval_log_few-shot_openai.csv" and g["tightest_diagonal_count"] == 4
    errors = g["total_errors_by_pathway"]
    assert errors["Carding_to_BIN_Testing"] == 3 and errors["BIN_Testing_to_Carding"] == 1
    assert errors["Stolen_Card_to_Legitimate"] == 1 and errors["Synthetic_Identity_to_Legitimate"] == 1
    assert sum(errors.values()) == 6  # Drop_House is outside the 5 categories
    assert {"true": "Legitimate", "pred": "Carding"} in g["zero_error_cells"]
    assert {"true": "Carding", "pred": "BIN_Testing"} not in g["zero_error_cells"]

    # A missing log must stop the run, not silently shrink the analysis
    os.remove(os.path.join(tmp, "llm_eval_log_zero-shot_anthropic.csv"))
    os.remove(os.path.join(tmp, "confusion_matrix_analysis.json"))
    r = run(tmp, "build_confusion_matrix.py")
    assert r.returncode != 0, "missing log did not fail the run"
    assert not os.path.exists(os.path.join(tmp, "confusion_matrix_analysis.json"))


def v4_performance_grid():
    tmp = fixture_dir("cost-accuracy_analysis.py")
    r = run(tmp, "cost-accuracy_analysis.py")
    assert r.returncode == 0, r.stdout + r.stderr
    grid = json.load(open(os.path.join(tmp, "performance_grid_analysis.json")))
    expected_acc = {"zero-shot": {"openai": 25.0, "anthropic": 75.0}, "few-shot": {"openai": 100.0, "anthropic": 75.0},
                    "cot": {"openai": 75.0, "anthropic": 75.0}}
    for s in STRATEGIES:
        for e in ENGINES:
            cell = grid[s][e]
            assert cell["status"] == "OK", (s, e)
            assert abs(cell["accuracy"] - expected_acc[s][e]) < 1e-9, (s, e, cell)
            k = list(FIXTURE).index(f"{s}_{e}") + 1
            assert abs(cell["latency"] - 300.0 * k) < 1e-9, (s, e, cell)
            assert abs(cell["cost"] - 0.010 * k) < 1e-12, (s, e, cell)


def load_cot_module():
    tmp = tempfile.mkdtemp()  # no .env here, so the dummy keys below are used
    path = shutil.copy(os.path.join(ROOT, "classify_llm_chain-of-thought.py"), tmp)
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
    spec = importlib.util.spec_from_file_location("cot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def v5_llm_scripts():
    cot = load_cot_module()
    assert cot.extract_classification("<scratchpad>CVV matched</scratchpad>\n<classification>\n Carding \n</classification>") == "Carding"
    assert cot.extract_classification("<CLASSIFICATION>BIN_Testing</CLASSIFICATION>") == "BIN_Testing"
    assert cot.extract_classification("Final answer: Carding") == "ERROR_PARSING"
    # Same method for every run: fixed sample file, temperature 0 on both engines
    for script in glob.glob(os.path.join(ROOT, "classify_llm_*.py")):
        src = open(script).read()
        assert len(re.findall(r'^\s+temperature=0\.0,', src, re.M)) == 1 \
            and len(re.findall(r'^\s+extra_body=\{"temperature": 0\.0\},', src, re.M)) == 1, \
            f"{script}: temperature 0 not set on both engines"
        assert "eval_sample_ids.txt" in src, f"{script}: does not use the shared sample"
        assert "random.sample" not in src, f"{script}: draws its own random sample"


def v6_same_sample():
    ids = open(os.path.join(ROOT, "eval_sample_ids.txt")).read().split()
    assert len(ids) == 50 and len(set(ids)) == 50, "need 50 unique ids"
    dataset = json.load(open(os.path.join(ROOT, "synthetic_fraud_dataset.json")))
    labels = {r["raw_ml_features"]["transaction_id"]: r["raw_ml_features"]["fraud_category"] for r in dataset}
    assert Counter(labels[i] for i in ids) == Counter({c: 10 for c in CATEGORIES})
    for s in STRATEGIES:
        for e in ENGINES:
            f = os.path.join(ROOT, f"llm_eval_log_{s}_{e}.csv")
            rows = list(csv.DictReader(open(f)))
            assert [r["transaction_id"] for r in rows] == ids, f"{f}: not the shared sample"
            assert all(r["true_label"] == labels[r["transaction_id"]] for r in rows), f"{f}: wrong true labels"


def v7_narratives():
    import dataset_generator as g
    n = g.enrich_for_llm(raw_record(amount=1.0, transaction_type="Pre-Auth", velocity_pan_count_1m_ip=20,
                                    velocity_cvv_fail_count_5m_bin=30, cvv_result="Mismatch",
                                    auth_response_code="N7 - CVV Mismatch", account_age_days=0,
                                    billing_zip="00000", shipping_zip="00000",
                                    field_input_method="Programmatic Inject", session_duration_ms=300,
                                    proxy_vpn_flag=True)["raw_ml_features"])["llm_reasoning_prompt"]
    assert n["amount_and_intent"].startswith("A USD 1.00 pre-authorization request")
    assert n["verification_outcome"] == "Authorization failed due to a CVV Mismatch (Code N7 - CVV Mismatch)."
    assert n["velocity_context"].startswith("Extreme velocity anomaly: This IP has attempted 20 distinct card numbers")
    assert n["identity_profile"].startswith("This is a brand-new guest checkout session")
    assert n["logistics_context"].startswith("No physical delivery address provided")
    assert n["behavior_and_network"].startswith("Severe bot signature: Checkout completed in an impossible 300 milliseconds")

    n = g.enrich_for_llm(raw_record(account_age_days=5, identity_synthetic_score=0.9, email_domain="temp-mail.org",
                                    shipping_zip="94105", auth_response_code="51 - Insufficient Funds",
                                    velocity_pan_count_1m_ip=6)["raw_ml_features"])["llm_reasoning_prompt"]
    assert n["identity_profile"].startswith("Account is highly suspicious. Created 5 days ago")
    assert n["logistics_context"].startswith("High logistics risk: Billing ZIP (10001)")
    assert n["verification_outcome"].startswith("CVV matched, but the transaction was declined")
    assert n["velocity_context"] == "High velocity: This session has rotated through 6 different cards in under a minute."


def v8_reports_match_data():
    # Grid recomputed here from the logs, without the project's scripts
    grid = {}
    for s in STRATEGIES:
        for e in ENGINES:
            rows = list(csv.DictReader(open(os.path.join(ROOT, f"llm_eval_log_{s}_{e}.csv"))))
            grid[(s, e)] = {
                "accuracy": 100 * sum(r["true_label"] == r["predicted_label"] for r in rows) / len(rows),
                "latency": sum(float(r["latency_ms"]) for r in rows) / len(rows),
                "cost": sum(float(r["cost_usd"]) for r in rows),
            }
    saved = json.load(open(os.path.join(ROOT, "performance_grid_analysis.json")))
    for (s, e), cell in grid.items():
        for k, v in cell.items():
            assert abs(saved[s][e][k] - v) < 1e-6, f"performance_grid_analysis.json is stale: {s}/{e}/{k}"

    vp = open(os.path.join(ROOT, "VP_Report.md")).read()
    pct = lambda s, e: f"{grid[(s, e)]['accuracy']:.0f}%"
    assert f"from {pct('zero-shot', 'openai')} to {pct('few-shot', 'openai')} on GPT-4o-mini" in vp
    assert f"from {pct('zero-shot', 'anthropic')} to {pct('few-shot', 'anthropic')} on Claude Haiku 4.5" in vp
    assert f"({pct('cot', 'openai')} on GPT-4o-mini, {pct('cot', 'anthropic')} on Claude Haiku 4.5)" in vp
    fastest = min(grid.values(), key=lambda c: c["latency"])["latency"]
    assert f"averaged {fastest:.0f}ms per transaction" in vp
    slowest_cot = max(grid[("cot", e)]["latency"] for e in ENGINES)
    assert f"as much as {slowest_cot:.0f}ms" in vp
    for e in ENGINES:  # the report says CoT is less accurate than few-shot
        assert grid[("cot", e)]["accuracy"] < grid[("few-shot", e)]["accuracy"], e

    readme = open(os.path.join(ROOT, "README.md")).read()
    for s, label in [("zero-shot", "Zero-shot"), ("few-shot", "Few-shot (2 examples)"), ("cot", "Chain-of-thought")]:
        cells = [f"{grid[(s, e)]['accuracy']:.0f}% · {grid[(s, e)]['latency']:,.0f} ms · ${grid[(s, e)]['cost']:.4f}" for e in ENGINES]
        row = re.search(rf"^\| {re.escape(label)} \| (.+) \| (.+) \|$", readme, re.M)
        assert row, f"README row missing: {label}"
        assert [c.replace("**", "") for c in row.groups()] == cells, f"README row {label}: {row.groups()} != {cells}"
    rule_rows = list(csv.DictReader(open(os.path.join(ROOT, "rule_engine_eval_log.csv"))))
    ids = set(open(os.path.join(ROOT, "eval_sample_ids.txt")).read().split())
    hits = sum(r["is_correct"] == "True" for r in rule_rows if r["transaction_id"] in ids)
    assert f"| Rule engine (baseline) | {hits}/50 = {hits * 2}% ·" in readme
    overall = 100 * sum(r["is_correct"] == "True" for r in rule_rows) / len(rule_rows)
    assert f"the rule engine scores {overall:.1f}%" in readme

    # Committed outputs are current: rule log matches the engine, matrices match the logs
    import card_fraud_analyzer as c
    dataset = c.load_dataset(os.path.join(ROOT, "synthetic_fraud_dataset.json"))
    expected = [c.rule_based_fallback_classifier(c.extract_features(r)) for r in dataset]
    assert [r["predicted_label"] for r in rule_rows] == expected, "rule_engine_eval_log.csv is stale"
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(ROOT, "build_confusion_matrix.py"), tmp)
    for s in STRATEGIES:
        for e in ENGINES:
            shutil.copy(os.path.join(ROOT, f"llm_eval_log_{s}_{e}.csv"), tmp)
    assert run(tmp, "build_confusion_matrix.py").returncode == 0
    assert json.load(open(os.path.join(tmp, "confusion_matrix_analysis.json"))) == \
        json.load(open(os.path.join(ROOT, "confusion_matrix_analysis.json"))), "confusion_matrix_analysis.json is stale"

    # Every cited transaction is a CoT error with the stated labels, and no error is left out
    report = open(os.path.join(ROOT, "Engineer_Report.md")).read()
    sections = {"anthropic": report.split("## Claude Haiku 4.5")[1].split("## GPT-4o-mini")[0],
                "openai": report.split("## GPT-4o-mini")[1]}
    for e, text in sections.items():
        rows = {r["transaction_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, f"llm_eval_log_cot_{e}.csv")))}
        cited = set()
        for true_label, pred_label, n, ids in re.findall(
                r"### (\w+) predicted as (\w+) \((\d+)\)\n\* \*\*Transactions:\*\* ([^\n]+)", text):
            ids = re.findall(r"tx_[0-9a-f]{10}", ids)
            assert len(ids) == int(n), f"{e} {true_label}->{pred_label}: count {n} but {len(ids)} ids"
            for i in ids:
                assert i in rows, f"{i} not in the {e} CoT log"
                assert (rows[i]["true_label"], rows[i]["predicted_label"]) == (true_label, pred_label), f"{e} {i}"
                cited.add(i)
        errors = {i for i, r in rows.items() if r["true_label"] != r["predicted_label"]}
        assert cited == errors, f"{e}: errors missing from the report: {sorted(errors - cited)}"
    assert set(re.findall(r"tx_[0-9a-f]{10}", report)) <= set(re.findall(r"tx_[0-9a-f]{10}", "".join(sections.values())))

    # Regenerating the reports from the committed data changes nothing
    tmp = tempfile.mkdtemp()
    for f in ["generate_reports.py", "performance_grid_analysis.json", "llm_eval_log_cot_openai.csv",
              "llm_eval_log_cot_anthropic.csv"]:
        shutil.copy(os.path.join(ROOT, f), tmp)
    r = run(tmp, "generate_reports.py")
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ["VP_Report.md", "Engineer_Report.md"]:
        assert open(os.path.join(tmp, f)).read() == open(os.path.join(ROOT, f)).read(), f"{f} is stale"


CHECKS = {"V1": v1_no_secrets, "V2": v2_rule_engine, "V3": v3_confusion_matrix, "V4": v4_performance_grid,
          "V5": v5_llm_scripts, "V6": v6_same_sample, "V7": v7_narratives, "V8": v8_reports_match_data}

if __name__ == "__main__":
    CHECKS[sys.argv[1]]()
