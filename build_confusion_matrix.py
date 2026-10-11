"""
Count which labels each model run mixes up.

For each of the six main result logs it builds a confusion matrix: a table
where rows are the true labels and columns are what the model answered.
Numbers on the diagonal are right answers; every other cell is a specific
mistake, such as 8 carding payments answered as BIN testing.

It also reports the most one-sided mix-up in each run and the label pairs that
no run ever confused. Results go to confusion_matrix_analysis.json.

How to run (from the project folder): python3 build_confusion_matrix.py
"""
import os
import sys
import csv
import json

files_to_analyze = [
    "llm_eval_log_zero-shot_openai.csv",
    "llm_eval_log_zero-shot_anthropic.csv",
    "llm_eval_log_few-shot_openai.csv",
    "llm_eval_log_few-shot_anthropic.csv",
    "llm_eval_log_cot_openai.csv",
    "llm_eval_log_cot_anthropic.csv"
]

base_categories = [
    "Legitimate",
    "BIN_Testing",
    "Carding",
    "Stolen_Card",
    "Synthetic_Identity"
]

# Trackers for global analysis
best_diagonal_count = -1
best_file_name = ""

# Misclassification counts per (true, predicted) pair, summed over all files
global_off_diagonal_counts = {}
for t in base_categories:
    for p in base_categories:
        if t != p:
            global_off_diagonal_counts[f"{t}_to_{p}"] = 0

# Saved to JSON at the end so later scripts can load it
output_data = {
    "per_file_analysis": {},
    "global_summary": {}
}

# Per-file confusion matrices
for filename in files_to_analyze:
    if not os.path.exists(filename):
        sys.exit(f"Could not find {filename}. Run the matching classify_llm_* script first.")

    current_categories = base_categories.copy()

    # 1. Empty matrix
    confusion_matrix = {}
    for true_cat in current_categories:
        confusion_matrix[true_cat] = {}
        for pred_cat in current_categories:
            confusion_matrix[true_cat][pred_cat] = 0

    total_records = 0
    correct_predictions = 0

    # 2. Count predictions
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row["true_label"]
            pred_label = row["predicted_label"]

            # CoT runs can return labels outside the 5 categories; give them their own row/column
            if true_label not in confusion_matrix:
                confusion_matrix[true_label] = {}
                current_categories.append(true_label)
                for existing_pred in current_categories:
                    confusion_matrix[true_label][existing_pred] = 0

            if pred_label not in current_categories:
                current_categories.append(pred_label)
                for t_cat in confusion_matrix.keys():
                    confusion_matrix[t_cat][pred_label] = 0

            confusion_matrix[true_label][pred_label] += 1
            total_records += 1
            if true_label == pred_label:
                correct_predictions += 1

            # Error counts across all files, used to find cells that are never wrong
            if true_label in base_categories and pred_label in base_categories:
                if true_label != pred_label:
                    global_off_diagonal_counts[f"{true_label}_to_{pred_label}"] += 1

    # Track the file with the most correct predictions
    if correct_predictions > best_diagonal_count:
        best_diagonal_count = correct_predictions
        best_file_name = filename

    # 3. Print the matrix
    print(f"{'='*115}")
    print(f"CONFUSION MATRIX: {filename.upper()}")
    print(f"{'='*115}")
    print("ROWS = True Labels | COLUMNS = Predicted Labels\n")

    col_width = 20
    header_row = "".ljust(col_width)
    for cat in current_categories:
        header_row += cat.ljust(col_width)
    print(header_row)
    print("-" * len(header_row))

    for true_cat in current_categories:
        if true_cat not in confusion_matrix:
            continue
        row_string = true_cat.ljust(col_width)
        for pred_cat in current_categories:
            count = confusion_matrix[true_cat].get(pred_cat, 0)
            row_string += str(count).ljust(col_width)
        print(row_string)

    # 4. Most lopsided confusion in this file
    accuracy = (correct_predictions / total_records) * 100 if total_records > 0 else 0
    print(f"\n>> ACCURACY / DIAGONAL: {accuracy:.1f}% ({correct_predictions}/{total_records})")

    max_asym_diff = 0
    asym_a = ""
    asym_b = ""
    a_to_b_val = 0
    b_to_a_val = 0

    for i in range(len(current_categories)):
        for j in range(i + 1, len(current_categories)):
            cat_a = current_categories[i]
            cat_b = current_categories[j]

            a_to_b = confusion_matrix.get(cat_a, {}).get(cat_b, 0)
            b_to_a = confusion_matrix.get(cat_b, {}).get(cat_a, 0)  # predicted-only labels have no row
            diff = abs(a_to_b - b_to_a)

            if diff > max_asym_diff:
                max_asym_diff = diff
                if a_to_b > b_to_a:
                    asym_a, asym_b = cat_a, cat_b
                    a_to_b_val, b_to_a_val = a_to_b, b_to_a
                else:
                    asym_a, asym_b = cat_b, cat_a
                    a_to_b_val, b_to_a_val = b_to_a, a_to_b

    if max_asym_diff > 0:
        print(f">> GREATEST ASYMMETRIC CONFUSION: '{asym_a}' misclassified as '{asym_b}' ({a_to_b_val} times), but '{asym_b}' misclassified as '{asym_a}' only ({b_to_a_val} times).")
    else:
        print(">> GREATEST ASYMMETRIC CONFUSION: None (errors are symmetric or zero).")

    print("\n\n")

    # 5. Store for the JSON output
    output_data["per_file_analysis"][filename] = {
        "total_records": total_records,
        "correct_predictions": correct_predictions,
        "accuracy_percent": round(accuracy, 2),
        "raw_matrix": confusion_matrix,
        "greatest_asymmetric_confusion": {
            "true_label": asym_a,
            "predicted_label": asym_b,
            "count_a_to_b": a_to_b_val,
            "count_b_to_a": b_to_a_val,
            "difference": max_asym_diff
        } if max_asym_diff > 0 else None
    }


# Comparison across all runs
print(f"{'='*115}")
print("ACROSS ALL RUNS")
print(f"{'='*115}\n")

# 1. Tightest Diagonal
print("1. Most correct answers:")
print(f"   Winner: {best_file_name}")
print(f"   Score: {best_diagonal_count} correct classifications on the main diagonal.\n")

# 2. Off-diagonal cells with zero errors in every run
print("2. Label pairs no run ever confused:")
zero_error_pairs = []
for label_pair, err_sum in global_off_diagonal_counts.items():
    if err_sum == 0:
        t, p = label_pair.split("_to_")
        zero_error_pairs.append({"true": t, "pred": p})

if zero_error_pairs:
    for pair in zero_error_pairs:
        print(f"   • True: '{pair['true']}'  -->  Predicted: '{pair['pred']}' (0 errors across all logs)")
else:
    print("   • None (every mix-up happened at least once).")

# Populate global summary for JSON
output_data["global_summary"] = {
    "tightest_diagonal_winner": best_file_name,
    "tightest_diagonal_count": best_diagonal_count,
    "zero_error_cells": zero_error_pairs,
    "total_errors_by_pathway": global_off_diagonal_counts
}

output_filename = "confusion_matrix_analysis.json"
with open(output_filename, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"\nStructured analysis data saved to {output_filename}")