"""
Run in the same folder as results.jsonl:
    python filter_failures.py
"""
import json

path = "results.jsonl"
task_filter = "change_color"
max_print = 10

failures = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("task") != task_filter:
            continue
        if rec.get("metrics", {}).get("gold_patch_exact"):
            continue  # this one passed, skip
        failures.append(rec)

print(f"Total {task_filter} failures: {len(failures)}\n")

for rec in failures[:max_print]:
    print("=" * 70)
    print("case_id:", rec.get("case_id"))
    print("error:", rec.get("error"))
    print("metrics:", json.dumps(rec.get("metrics")))
    print("patch (model's actual output):")
    print(json.dumps(rec.get("patch"), indent=2))
    print()
