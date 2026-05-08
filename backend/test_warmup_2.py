import sys
from search_api.main import load_ma20_parquet, dict_to_matrix
from search_api.data_io import DATA_DIR
print("DATA_DIR:", DATA_DIR)
print("ma20.parquet exists:", (DATA_DIR / "ma20.parquet").exists())
df = load_ma20_parquet()
print("DF is None?", df is None)
if df is not None:
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
