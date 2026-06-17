import json
import glob
from pathlib import Path

cache_files = glob.glob("backend/data/cache_*.json")
for f in sorted(cache_files):
    print(f"\n--- {f} ---")
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if data:
                top5 = [f"{d['symbol']} ({d['prob']})" for d in data[:5]]
                print(", ".join(top5))
            else:
                print("Empty")
    except Exception as e:
        print("Error:", e)
