import json
with open("results.jsonl") as f:
    rec = json.loads(f.readline())
print("KEYS:", list(rec.keys()))
print()
print(json.dumps(rec, indent=2)[:1500])
