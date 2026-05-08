import sys
sys.path.append('.')
from backend.search_api.main import load_ma20_parquet, dict_to_matrix
df = load_ma20_parquet()
print("DF shape:", df.shape)
if 'ticker' in df.columns and 'vector' in df.columns:
    print("New format")
else:
    print("Old format")
    ma20 = {c: df[c].dropna() for c in df.columns}
    print("ma20 keys:", len(ma20.keys()))
    matrix, T = dict_to_matrix(ma20, target_len=128)
    print("Matrix shape:", matrix.shape)
    print("T len:", len(T))
