"""
Run from the repo root, same as before:
    python check_value_mismatch.py

For failures where the model's targets already match gold exactly, print
the model's attribute value vs gold's -- to confirm the bottleneck is the
color/attribute VALUE, not which node was picked.
"""
import json
from svgpatchlab.core.xml import index_tree, parse_svg
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.data.svgeditbench import SVGEditBench

RESULTS_PATH = "results.jsonl"
BENCH_ROOT = "SVGEditBench"
TASK_FILTER = {"change_color", "set_contour"}


def load_failures():
    failures = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("task") not in TASK_FILTER:
                continue
            if rec.get("metrics", {}).get("gold_patch_exact"):
                continue
            if rec.get("error") or not rec.get("patch"):
                continue
            failures.append(rec)
    return failures


def main():
    failures = load_failures()
    bench = SVGEditBench(BENCH_ROOT)
    cases_by_id = {f"{c.task}/{c.emoji_id}": c for c in bench.iter_cases(tasks=list(TASK_FILTER))}

    target_match_count = 0
    value_mismatch_count = 0

    for rec in failures:
        case_id = rec["case_id"]
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        try:
            gold_patch = derive_patch(case.source_svg, case.answer_svg)
        except Exception:
            continue

        gold_targets = set()
        gold_attrs = {}
        for op in gold_patch.operations:
            gold_targets.update(op.targets)
            gold_attrs.update(op.attributes_dict)

        model_ops = rec["patch"]["operations"]
        if not model_ops:
            continue
        predicted_targets = set(model_ops[0].get("targets", []))
        model_attrs = model_ops[0].get("attributes", {})

        if predicted_targets != gold_targets:
            continue  # targeting mismatch, not what we're checking here

        target_match_count += 1
        if model_attrs != gold_attrs:
            value_mismatch_count += 1
            print(f"{case_id}: model_attrs={model_attrs}  gold_attrs={gold_attrs}")

    print(f"\n{value_mismatch_count}/{target_match_count} target-matching failures "
          f"had a wrong attribute VALUE")


if __name__ == "__main__":
    main()
