import requests
import numpy as np

# Create a simple sketch
y = np.linspace(1, 0, 128).tolist()

payload_1y = {"y": y, "target_len": 128, "period": "1y"}
payload_2y = {"y": y, "target_len": 128, "period": "2y"}
payload_3m = {"y": y, "target_len": 128, "period": "3m"}

url = "https://drs-recommend-service-405087545852.asia-northeast3.run.app/api/search/similar"
try:
    r1 = requests.post(url, json=payload_1y).json()
    r2 = requests.post(url, json=payload_2y).json()
    r3 = requests.post(url, json=payload_3m).json()
except Exception as e:
    print("URL might be incorrect:", e)
    import sys
    sys.exit(1)

def print_res(name, data):
    print(f"--- {name} ---")
    if "items" in data:
        for it in data["items"]:
            print(f"{it['ticker']}: {it['score']:.4f}")
    else:
        print(data)

print_res("1y", r1)
print_res("2y", r2)
print_res("3m", r3)
