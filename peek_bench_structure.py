from pathlib import Path
p = Path("SVGEditBench/1_ChangeColor")
print("Contents of", p, ":")
for item in sorted(p.iterdir())[:15]:
    print(" ", item.name, "(dir)" if item.is_dir() else "")
