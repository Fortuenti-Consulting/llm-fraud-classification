import os
import csv
import json

strategies = ["zero-shot", "few-shot", "cot"]
engines = ["openai", "anthropic"]

grid = {}
for strat in strategies:
    grid[strat] = {}
    for eng in engines:
        grid[strat][eng] = {
            "accuracy": 0.0,
            "latency": 0.0,
            "cost": 0.0,
            "status": "Missing"
        }

for strat in strategies:
    for eng in engines:
        filename = f"llm_eval_log_{strat}_{eng}.csv"

        if not os.path.exists(filename):
            continue

        total_records = 0
        correct_predictions = 0
        total_latency = 0.0
        total_cost = 0.0

        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_records += 1
                if row.get("is_correct") == "True":
                    correct_predictions += 1

                total_latency += float(row.get("latency_ms", 0.0))
                total_cost += float(row.get("cost_usd", 0.0))

        if total_records > 0:
            grid[strat][eng]["accuracy"] = (correct_predictions / total_records) * 100
            grid[strat][eng]["latency"] = total_latency / total_records
            grid[strat][eng]["cost"] = total_cost
            grid[strat][eng]["status"] = "OK"

print("\n" + "=" * 90)
print(f"{'LLM ARCHITECTURE PERFORMANCE GRID':^90}")
print("=" * 90)

print(f"{'':15} | {'OPENAI (GPT-4o-mini)':^33} | {'ANTHROPIC (Claude Haiku 4.5)':^33}")
print("-" * 90)

for strat in strategies:
    row_label = strat.upper()
    row_output = f"{row_label:15} |"

    for eng in engines:
        data = grid[strat][eng]
        if data["status"] == "OK":
            acc_str = f"Acc: {data['accuracy']:.1f}%"
            lat_str = f"Lat: {int(data['latency'])}ms"
            cost_str = f"${data['cost']:.4f}"
            cell_str = f" {acc_str} | {lat_str:10} | {cost_str:7} "
            row_output += f"{cell_str}|"
        else:
            row_output += f"{'Missing Data':^35}|"

    print(row_output)
    if strat != strategies[-1]:
        print("-" * 90)

print("=" * 90 + "\n")

# Recommendation derived from the grid: most accurate cell, cheapest cell within
# 5 points of it, and the strategy with the lowest mean accuracy across engines
names = {"openai": "GPT-4o-mini", "anthropic": "Claude Haiku 4.5"}
cells = [(s, e, grid[s][e]) for s in strategies for e in engines if grid[s][e]["status"] == "OK"]
if cells:
    top_s, top_e, top = max(cells, key=lambda c: c[2]["accuracy"])
    eco_s, eco_e, eco = min((c for c in cells if c[2]["accuracy"] >= top["accuracy"] - 5), key=lambda c: c[2]["cost"])
    worst = min(strategies, key=lambda s: sum(grid[s][e]["accuracy"] for e in engines))
    print(">> ARCHITECTURE RECOMMENDATION:")
    print(f"   - For MAX ACCURACY:    {names[top_e]} + {top_s} ({top['accuracy']:.0f}%)")
    print(f"   - For MAX ECONOMY:     {names[eco_e]} + {eco_s} ({eco['accuracy']:.0f}%, ${eco['cost']:.4f} per 50)")
    print(f"   - AVOID:               {worst} (lowest mean accuracy across engines)\n")

output_filename = "performance_grid_analysis.json"
with open(output_filename, "w", encoding="utf-8") as json_file:
    json.dump(grid, json_file, indent=4)

print(f"Performance grid data saved to {output_filename}\n")