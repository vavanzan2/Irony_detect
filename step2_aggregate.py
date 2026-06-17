import os
import pandas as pd


def detect_columns(df):
    """Detect the names of relevant columns in the raw DataFrame."""
    cols = df.columns.tolist()

    def find_col(candidates):
        for c in candidates:
            if c in cols:
                return c
        for c in cols:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        return None

    item_id  = find_col(["item_id", "id_original", "text_id"])
    label    = find_col(["label", "irony", "annotation"])
    reply    = find_col(["text", "reply"])
    post     = find_col(["parent_text", "post"])
    platform = find_col(["source", "platform"])

    detected = {
        "item_id": item_id,
        "label":   label,
        "reply":   reply,
        "post":    post,
        "platform": platform,
    }
    print(f"  Detected columns: {detected}")
    return detected


def encode_label(value):
    """Convert an irony label value to int (1=ironic, 0=not ironic).
    Covers EPIC's actual values: 'iro' / 'not'.
    """
    if isinstance(value, (int, float)):
        return int(bool(value))
    v = str(value).strip().lower()
    if v in ("iro", "irony", "ironic", "1", "true", "yes"):
        return 1
    return 0


def majority_vote(df, cols):
    """Aggregate annotations by item_id using majority vote.
    Returns a DataFrame with one row per text.
    """
    print("[Step 2] Applying majority vote...")

    id_col    = cols["item_id"]
    label_col = cols["label"]
    reply_col = cols["reply"]
    post_col  = cols["post"]
    plat_col  = cols["platform"]

    df = df.copy()
    df["_label_int"] = df[label_col].apply(encode_label)

    votes = (
        df.groupby(id_col) ["_label_int"]
        .agg(vote_ironic="sum", n_annotators="count")
        .reset_index()
    )
    votes.rename(columns={id_col: "item_id"}, inplace=True)
    votes["label"] = (votes["vote_ironic"] >= votes["n_annotators"] / 2).astype(int)

    first_src = {}
    if reply_col:
        first_src[reply_col] = "text"
    if post_col:
        first_src[post_col] = "post"
    if plat_col:
        first_src[plat_col] = "platform"

    if first_src:
        text_df = (
            df.groupby(id_col)[list(first_src.keys())]
            .first()
            .reset_index()
        )
        text_df.rename(columns={id_col: "item_id", **{k: v for k, v in first_src.items()}}, inplace=True)
        labeled = votes.merge(text_df, on="item_id")
    else:
        labeled = votes

    print(f"  Unique texts: {len(labeled)}")
    print(f"  Final distribution:\n{labeled['label'].value_counts().to_string()}")
    print(f"  (1 = ironic, 0 = not ironic)")
    return labeled


def save_labeled(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "epic_labeled.csv")
    df.to_csv(path, index=False)
    print(f"\n[Step 2] Labeled dataset saved at: {path}")
    return path


def run(raw_df=None, output_dir="outputs"):
    if raw_df is None:
        path = os.path.join(output_dir, "epic_raw.csv")
        print(f"[Step 2] Loading {path}...")
        raw_df = pd.read_csv(path)

    cols = detect_columns(raw_df)
    labeled_df = majority_vote(raw_df, cols)
    save_labeled(labeled_df, output_dir)
    print("\n[Step 2] Completed successfully.")
    return labeled_df
