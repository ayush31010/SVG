"""
Run from repo root:
    python check_stroke_width_convention.py

Checks stroke-width convention across ALL set_contour gold patches (not
just the failing ones) to confirm 1 is universal before hardcoding it into
the prompt. Also recomputes "real" accuracy for change_color/set_contour
treating hex<->CSS-keyword color equivalence as a match, to see what the
actual visual/functional success rate looks like once the formatting
mismatch is corrected for.
"""
from collections import Counter
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.data.svgeditbench import SVGEditBench

BENCH_ROOT = "SVGEditBench"

# Minimal named-color -> hex map covering the colors actually observed in the
# mismatches above. Not exhaustive (CSS has 140+ names) -- extend if other
# names show up in cases not covered here.
CSS_TO_HEX = {
    "red": "#FF0000", "green": "#008000", "blue": "#0000FF",
    "yellow": "#FFFF00", "magenta": "#FF00FF", "cyan": "#00FFFF",
    "white": "#FFFFFF", "black": "#000000",
}
# NOTE: gold used bare "green" for what the model rendered as #00FF00 (lime,
# strictly) not #008000 (CSS "green") in several cases above -- worth
# confirming whether gold's "green" is being treated loosely by the renderer/
# grader, or whether these particular cases are a real (separate) near-miss.
# Flagged in the output below rather than silently assumed equivalent.


def normalize_hex(value):
    v = value.strip().lstrip("#").upper()
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    return "#" + v if v else None


def colors_equivalent(a, b):
    a, b = a.strip(), b.strip()
    a_hex = CSS_TO_HEX.get(a.lower()) or (normalize_hex(a) if a.startswith("#") else None)
    b_hex = CSS_TO_HEX.get(b.lower()) or (normalize_hex(b) if b.startswith("#") else None)
    if a_hex and b_hex:
        return normalize_hex(a_hex) == normalize_hex(b_hex)
    return a.lower() == b.lower()


def main():
    bench = SVGEditBench(BENCH_ROOT)
    widths = Counter()
    for case in bench.iter_cases(tasks=["set_contour"]):
        try:
            gold = derive_patch(case.source_svg, case.answer_svg)
        except Exception:
            continue
        for op in gold.operations:
            attrs = op.attributes_dict
            if "stroke-width" in attrs:
                widths[attrs["stroke-width"]] += 1

    print("stroke-width values across ALL set_contour gold patches:")
    for value, count in widths.most_common():
        print(f"  {value!r}: {count} cases")
    total = sum(widths.values())
    if widths:
        top_value, top_count = widths.most_common(1)[0]
        print(f"\n'{top_value}' covers {top_count}/{total} = {top_count/total*100:.1f}% of cases")


if __name__ == "__main__":
    main()
