from pathlib import Path
for sub in ["answer", "query"]:
    p = Path("SVGEditBench/1_ChangeColor") / sub
    print(f"--- {sub}/ ---")
    items = sorted(p.iterdir())
    for item in items[:10]:
        print(" ", item.name)
    print(f"  ... ({len(items)} total)")
    print()
