import os
import numpy as np
import pandas as pd


def split_by_id(df, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split data by item_id to avoid leakage across splits.
    Returns (train_df, val_df, test_df).
    """
    ids = df["item_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train : n_train + n_val])
    test_ids  = set(ids[n_train + n_val :])

    train_df = df[df["item_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["item_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["item_id"].isin(test_ids)].reset_index(drop=True)

    print(f"[Step 3] Split completed:")
    print(f"  Train      : {len(train_df)} texts  ({len(train_df)/n*100:.1f}%)")
    print(f"  Validation : {len(val_df)} texts  ({len(val_df)/n*100:.1f}%)")
    print(f"  Test       : {len(test_df)} texts  ({len(test_df)/n*100:.1f}%)")

    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        dist = split["label"].value_counts(normalize=True)
        print(f"  {name} — ironic: {dist.get(1, 0)*100:.1f}%  not ironic: {dist.get(0, 0)*100:.1f}%")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")


def run(labeled_df=None, seed=42, output_dir="outputs"):
    if labeled_df is None:
        path = os.path.join(output_dir, "epic_labeled.csv")
        print(f"[Step 3] Loading {path}...")
        labeled_df = pd.read_csv(path)

    train_df, val_df, test_df = split_by_id(labeled_df, seed=seed)
    save_splits(train_df, val_df, test_df, output_dir)
    print("\n[Step 3] Completed successfully.")
    return train_df, val_df, test_df
