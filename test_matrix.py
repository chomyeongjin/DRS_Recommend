import pandas as pd
from backend.search_api.data_io import slice_series_dict
from backend.search_api.features import dict_to_matrix
import numpy as np

df = pd.read_parquet("backend/data/ma20.parquet")
ma20 = {c: df[c].dropna() for c in df.columns}

s1y = slice_series_dict(ma20, 365)
s2y = slice_series_dict(ma20, 730)

m1, t1 = dict_to_matrix(s1y, 128)
m2, t2 = dict_to_matrix(s2y, 128)

print("m1 shape:", m1.shape)
print("m2 shape:", m2.shape)

if m1.shape == m2.shape:
    diff = np.abs(m1 - m2).sum()
    print("Difference between m1 and m2:", diff)
    print("Are they identical?", np.allclose(m1, m2))
