"""
Run from repo root:
    python check_instruction_wording.py

Checks whether the gold color/attribute value is literally the same word
used in the instruction text -- confirming whether the model is "helpfully"
converting a named color from the instruction into hex, instead of just
echoing the literal word gold expects.
"""
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.data.svgeditbench import SVGEditBench

BENCH_ROOT = "SVGEditBench"
CASE_IDS = ["change_color/1f37b", "change_color/1f387", "change_color/1f3a9", "set_contour/1f199"]

bench = SVGEditBench(BENCH_ROOT)
cases = {f"{c.task}/{c.emoji_id}": c for c in bench.iter_cases()}

for case_id in CASE_IDS:
    case = cases.get(case_id)
    if not case:
        print(f"[skip] {case_id} not found")
        continue
    gold = derive_patch(case.source_svg, case.answer_svg)
    gold_attrs = {}
    for op in gold.operations:
        gold_attrs.update(op.attributes_dict)
    print("=" * 60)
    print("case_id:", case_id)
    print("instruction:", case.instruction[:300])
    print("gold_attrs:", gold_attrs)
    for value in gold_attrs.values():
        print(f"  -> literal word {value!r} appears in instruction:", value.lower() in case.instruction.lower())
    print()
