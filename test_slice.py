import pandas as pd
from backend.search_api.data_io import slice_series_dict
from pathlib import Path

df = pd.read_parquet("backend/data/ma20.parquet")
ma20 = {c: df[c].dropna() for c in df.columns}
print("Total tickers:", len(ma20))

s3m = slice_series_dict(ma20, 90)
s1y = slice_series_dict(ma20, 365)
s2y = slice_series_dict(ma20, 730)

t = list(ma20.keys())[0]
print("Ticker:", t)
print("3m len:", len(s3m[t]))
print("1y len:", len(s1y[t]))
print("2y len:", len(s2y[t]))
