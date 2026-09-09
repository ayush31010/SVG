"""
Run from inside the repo root (where svgpatchlab/ and SVGEditBench/ both exist):
    python test_same_fill_theory.py

Tests the same-fill-grouping theory cheaply, no model calls: for each
change_color/set_contour failure in results.jsonl, expand the model's
predicted target(s) to include every other node that shared the EXACT
same original attribute value being changed, then check whether that
expanded set now matches the real gold target set.

Uses the real svgpatchlab.data.svgeditbench.SVGEditBench loader directly
(confirmed against the actual source: query/*.txt files are parsed via a
fenced ```svg block, not raw .svg files) rather than reading benchmark
files by hand.
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


def node_by_id(indexed_nodes):
    return {n.node_id: n for n in indexed_nodes}


def expand_with_same_value_siblings(predicted_targets, attr_name, original_nodes):
    by_id = node_by_id(original_nodes)
    original_values = set()
    for t in predicted_targets:
        node = by_id.get(t)
        if node is not None and attr_name in node.element.attrib:
            original_values.add(node.element.attrib[attr_name])
    if not original_values:
        return set(predicted_targets)
    expanded = set()
    for node in original_nodes:
        if node.element.attrib.get(attr_name) in original_values:
            expanded.add(node.node_id)
    return expanded


def main():
    failures = load_failures()
    print(f"Testing {len(failures)} change_color/set_contour failures\n")

    # Build a case_id -> BenchmarkCase lookup using the real loader
    bench = SVGEditBench(BENCH_ROOT)
    cases_by_id = {}
    for case in bench.iter_cases(tasks=list(TASK_FILTER)):
        case_id = f"{case.task}/{case.emoji_id}"
        cases_by_id[case_id] = case

    fixed_count = 0
    tested_count = 0

    for rec in failures:
        case_id = rec["case_id"]
        case = cases_by_id.get(case_id)
        if case is None:
            print(f"[skip] {case_id}: not found via loader")
            continue

        try:
            gold_patch = derive_patch(case.source_svg, case.answer_svg)
        except Exception as e:
            print(f"[skip] {case_id}: derive_patch failed: {e}")
            continue

        gold_targets = set()
        for op in gold_patch.operations:
            gold_targets.update(op.targets)

        model_ops = rec["patch"]["operations"]
        if not model_ops:
            continue
        predicted_targets = set(model_ops[0].get("targets", []))
        attrs_changed = model_ops[0].get("attributes", {})
        if not attrs_changed:
            continue
        attr_name = next(iter(attrs_changed))

        original_nodes = index_tree(parse_svg(case.source_svg))
        expanded = expand_with_same_value_siblings(predicted_targets, attr_name, original_nodes)

        tested_count += 1
        would_match = expanded == gold_targets
        if would_match:
            fixed_count += 1

        print(f"{case_id}: predicted={sorted(predicted_targets)} "
              f"expanded={sorted(expanded)} gold={sorted(gold_targets)} "
              f"{'FIXED' if would_match else 'still wrong'}")

    if tested_count:
        print(f"\n{fixed_count}/{tested_count} failures would be fixed by same-value expansion "
              f"({fixed_count/tested_count*100:.1f}%)")
    else:
        print("No cases tested -- check output above for why")


if __name__ == "__main__":
    main()
