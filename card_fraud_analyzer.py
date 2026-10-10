import json
import csv
import os
import sys
import time
from collections import Counter

DISPOSABLE_EMAIL_DOMAINS = {"temp-mail.org", "guerrillamail.com", "10minutemail.com", "throwawaymail.com"}

def load_dataset(dataset_path="synthetic_fraud_dataset.json"):
    """Loads the synthetic fraud dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_features(record):
    """
    Turns the raw fields of a dataset record into the tabular features
    the rule engine (or a future ML model) works with.
    """
    raw_ml = record["raw_ml_features"]
    auth_code = raw_ml["auth_response_code"].split(" - ")[0]  # "51 - Insufficient Funds" -> "51"

    features = {
        "transaction_id": raw_ml["transaction_id"],
        "fraud_category": raw_ml["fraud_category"],
        "amount": raw_ml["amount"],
        "cvv_match": raw_ml["cvv_result"] == "Match",
        "declined_code": None if auth_code == "00" else auth_code,
        "card_velocity_60s": raw_ml["velocity_pan_count_1m_ip"],
        "account_age_days": raw_ml["account_age_days"],
        "is_disposable_email": raw_ml["email_domain"] in DISPOSABLE_EMAIL_DOMAINS,
        "synthetic_identity_score": raw_ml["identity_synthetic_score"],
        "billing_shipping_zip_match": raw_ml["billing_zip"] == raw_ml["shipping_zip"],
        "is_proxy_or_datacenter": raw_ml["proxy_vpn_flag"],
        "checkout_latency_ms": raw_ml["session_duration_ms"]
    }
    return features

def rule_based_fallback_classifier(features):
    """
    Deterministic rule engine serving as a sub-5ms fallback/benchmark baseline.
    Rules follow the domain definitions worked out during prompt engineering.
    """
    # 1. BIN Testing: High velocity + explicit CVV mismatch
    if features["card_velocity_60s"] > 10 and not features["cvv_match"]:
        return "BIN_Testing"

    # 2. Carding: High velocity + matched CVV + issuer decline/insufficient funds
    if features["card_velocity_60s"] > 5 and features["cvv_match"] and features["declined_code"]:
        return "Carding"

    # 3. Synthetic Identity: High synthetic score or disposable email on new account
    if features["synthetic_identity_score"] > 0.80 or (features["is_disposable_email"] and features["account_age_days"] < 14):
        return "Synthetic_Identity"

    # 4. Stolen Card: Address mismatch on an established account (drop-house pattern)
    if not features["billing_shipping_zip_match"] and features["account_age_days"] > 30:
        return "Stolen_Card"

    # 5. Default Legitimate
    return "Legitimate"

def evaluate_dataset(dataset, output_csv="rule_engine_eval_log.csv"):
    """
    Runs the rule engine over the dataset, prints metrics and writes a per-record log.
    """
    results = []
    correct_count = 0
    total_latency_ms = 0.0

    print(f"Evaluating {len(dataset)} records with the rule engine...")

    for record in dataset:
        features = extract_features(record)
        true_label = features["fraud_category"]
        tx_id = features["transaction_id"]

        start_time = time.perf_counter()
        pred_label = rule_based_fallback_classifier(features)
        latency = (time.perf_counter() - start_time) * 1000

        is_correct = (pred_label == true_label)
        if is_correct:
            correct_count += 1

        total_latency_ms += latency

        results.append({
            "transaction_id": tx_id,
            "true_label": true_label,
            "predicted_label": pred_label,
            "is_correct": is_correct,
            "latency_ms": round(latency, 4)
        })

    accuracy = (correct_count / len(dataset)) * 100
    avg_latency = total_latency_ms / len(dataset)

    print("\n" + "="*45)
    print("RULE-BASED ANALYZER RESULTS")
    print("="*45)
    print(f"Total Records: {len(dataset)}")
    print(f"Accuracy:      {accuracy:.1f}% ({correct_count}/{len(dataset)})")
    print(f"Avg Latency:   {avg_latency:.4f} ms")

    # 92% of the dataset is Legitimate, so overall accuracy hides weak fraud
    # detection. Recall per category shows how each class is actually handled.
    totals = Counter(r["true_label"] for r in results)
    hits = Counter(r["true_label"] for r in results if r["is_correct"])
    print("\nRecall by category:")
    for category, total in totals.most_common():
        print(f"  {category:20} {hits[category] / total * 100:5.1f}% ({hits[category]}/{total})")
    print("="*45)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["transaction_id", "true_label", "predicted_label", "is_correct", "latency_ms"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation results saved to {output_csv}\n")
    return results

if __name__ == "__main__":
    try:
        dataset = load_dataset("synthetic_fraud_dataset.json")
        evaluate_dataset(dataset)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")
