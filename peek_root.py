from pathlib import Path
print("Current dir contents:")
for item in sorted(Path(".").iterdir()):
    print(" ", item.name, "(dir)" if item.is_dir() else "")

svgbench = Path("SVGEditBench")
print()
print("SVGEditBench exists:", svgbench.exists())
if svgbench.exists():
    print("SVGEditBench contents:")
    for item in sorted(svgbench.iterdir()):
        print(" ", item.name)
