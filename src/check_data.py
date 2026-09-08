import pandas as pd
from project_paths import UIT_VSMEC_RAW_DIR

train_path = UIT_VSMEC_RAW_DIR / "train.csv"
val_path = UIT_VSMEC_RAW_DIR / "validation.csv"
test_path = UIT_VSMEC_RAW_DIR / "test.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nColumns:")
print(train_df.columns)

print("\nFirst rows:")
print(train_df.head())

print("\nLabel distribution:")
print(train_df.iloc[:, -1].value_counts())