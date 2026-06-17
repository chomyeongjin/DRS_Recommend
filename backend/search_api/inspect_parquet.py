import pandas as pd
from pathlib import Path

p = Path("/Users/myeongjin/Documents/GitHub/DRS_Recommend/backend/search_api/data/ma20.parquet")
if p.exists():
    df = pd.read_parquet(p)
    print("Columns:", df.columns[:5])
    print("Index type:", type(df.index))
    if len(df) > 0:
        print("Min index:", df.index.min())
        print("Max index:", df.index.max())
    print("Shape:", df.shape)
else:
    print("File not found.")
