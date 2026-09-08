import pandas as pd
import matplotlib.pyplot as plt

from project_paths import UIT_VSMEC_RAW_DIR, FIGURES_DIR

TEXT_COL_CANDIDATES = ["Sentence", "sentence", "text", "Text", "comment", "Comment"]
LABEL_COL_CANDIDATES = ["Emotion", "emotion", "label", "Label"]

def infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Cannot infer text/label columns. Found: {list(df.columns)}")

    return text_col, label_col

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(UIT_VSMEC_RAW_DIR / "train.csv")
    val_df = pd.read_csv(UIT_VSMEC_RAW_DIR / "validation.csv")
    test_df = pd.read_csv(UIT_VSMEC_RAW_DIR / "test.csv")

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    text_col, label_col = infer_columns(df)

    df["num_words"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    label_counts = df[label_col].value_counts()
    label_counts.plot(kind="bar")
    plt.title("UIT-VSMEC Label Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=200)
    plt.close()

    df["num_words"].plot(kind="hist", bins=30)
    plt.title("Sentence Length Distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sentence_length_distribution.png", dpi=200)
    plt.close()

    print("Saved EDA figures to:", FIGURES_DIR)

if __name__ == "__main__":
    main()