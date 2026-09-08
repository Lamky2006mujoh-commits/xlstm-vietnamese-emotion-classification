from datasets import load_dataset
from project_paths import UIT_VSMEC_RAW_DIR

UIT_VSMEC_RAW_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("tridm/UIT-VSMEC")

for split_name, split_data in ds.items():
    out_path = UIT_VSMEC_RAW_DIR / f"{split_name}.csv"
    split_data.to_pandas().to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {split_name} to {out_path}")

print(ds)
