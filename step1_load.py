import os
import pandas as pd
from datasets import load_dataset


def load_epic_dataset():
    print("[Step 1] Downloading EPIC dataset from HuggingFace...")
    dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
    print(f"  Available splits: {list(dataset.keys())}")

    frames = []
    for split_name, split_data in dataset.items():
        df_split = split_data.to_pandas()
        df_split["original_split"] = split_name
        frames.append(df_split)

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded: {len(df)}")
    return df


def inspect_columns(df):
    print("\n[Step 1] Inspecting dataset columns...")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\n  Available columns:\n  {list(df.columns)}")
    print("\n  Data types:")
    print(df.dtypes.to_string(header=False))
    print("\n  Null values per column:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  No null values found.")
    print("\n  First 3 rows:")
    print(df.head(3).to_string())


def inspect_distributions(df):
    print("\n[Step 1] Analyzing distributions...")

    label_cols = [c for c in df.columns if "iron" in c.lower() or "label" in c.lower()]
    platform_cols = [c for c in df.columns if "platform" in c.lower() or "source" in c.lower()]
    annotator_cols = [c for c in df.columns if "annot" in c.lower() or "worker" in c.lower()]

    print(f"\n  Detected label columns:      {label_cols}")
    print(f"  Detected platform columns: {platform_cols}")
    print(f"  Detected annotator columns:   {annotator_cols}")

    for col in label_cols:
        print(f"\n  Distribution of '{col}':")
        print(df[col].value_counts(dropna=False).to_string())

    for col in platform_cols:
        print(f"\n  Distribution of '{col}':")
        print(df[col].value_counts(dropna=False).to_string())

    id_cols = [c for c in df.columns if "id" in c.lower() and "annot" not in c.lower()]
    for col in id_cols[:2]:
        print(f"\n  Unique values in '{col}': {df[col].nunique()}")

    for col in annotator_cols[:2]:
        print(f"  Unique values in '{col}': {df[col].nunique()}")


def save_raw_data(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "epic_raw.csv")
    df.to_csv(path, index=False)
    print(f"\n[Step 1] Raw dataset saved at: {path}")
    return path


def run(output_dir="outputs"):
    df = load_epic_dataset()
    inspect_columns(df)
    inspect_distributions(df)
    save_raw_data(df, output_dir)
    print("\n[Step 1] Completed successfully.")
    return df
