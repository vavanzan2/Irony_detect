import json
import os
import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline


CARDIFF_MODEL = "cardiffnlp/twitter-roberta-base-irony"


def load_test(output_dir):
    path = os.path.join(output_dir, "test.csv")
    df = pd.read_csv(path)
    print(f"[Step 5] Test loaded: {len(df)} texts")
    return df


def load_cardiff_pipeline():
    print(f"[Step 5] Downloading Cardiff model: {CARDIFF_MODEL}")
    pipe = pipeline(
        "text-classification",
        model=CARDIFF_MODEL,
        tokenizer=CARDIFF_MODEL,
        truncation=True,
        max_length=512,
    )
    print("  Model loaded.")
    return pipe


def detect_irony_label(pipe):
    """Discover which label string the model uses for irony and return a label→int mapping."""
    id2label = pipe.model.config.id2label
    print(f"  Cardiff id2label: {id2label}")

    # Split into tokens so negation markers are matched as whole words
    # (a plain substring check like "not" in n misses labels such as
    # "non_irony", since "non" does not contain "not" — that bug made
    # every label fall into the `else` branch and collapse to class 1).
    negation_tokens = {"not", "non", "no", "false", "0"}
    label_map = {}
    for idx, name in id2label.items():
        n = name.lower().strip()
        tokens = re.split(r"[^a-z0-9]+", n)
        if any(tok in negation_tokens for tok in tokens):
            label_map[name] = 0
        else:
            label_map[name] = 1

    print(f"  Applied mapping: {label_map}")
    return label_map


def predict_cardiff(pipe, texts, batch_size=32):
    """
    Run Cardiff in zero-shot mode and return binary predictions (1=ironic, 0=not ironic).
    Uses the model's id2label to ensure the correct mapping.
    """
    label_map = detect_irony_label(pipe)

    print(f"[Step 5] Predicting {len(texts)} texts in batches of {batch_size}...")
    outputs = pipe(list(texts), batch_size=batch_size, truncation=True, max_length=512)

    # Diagnostic: show first 3 raw predictions
    print(f"  Example raw outputs from the pipeline: {outputs[:3]}")

    preds = []
    for out in outputs:
        raw_label = out["label"]
        # Use detected mapping; fallback by string if not found
        if raw_label in label_map:
            preds.append(label_map[raw_label])
        else:
            label_str = raw_label.lower()
            if ("not" in label_str) or (label_str in ("label_0", "0")):
                preds.append(0)
            else:
                preds.append(1)

    return preds


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    report = classification_report(y_true, y_pred, target_names=["not ironic", "ironic"])

    results = {
        "model":     "Cardiff (zero-shot)",
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("\n[Step 5] Cardiff results (zero-shot):")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Macro-F1  : {results['macro_f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print("              Ironic  Not Ironic")
    print(f"  Ironic    {cm[0][0]:>7}  {cm[0][1]:>11}")
    print(f"  Not Ironic{cm[1][0]:>7}  {cm[1][1]:>11}")
    print(f"\n{report}")
    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cardiff_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Step 5] Results saved at: {path}")


def run(test_df=None, output_dir="outputs"):
    if test_df is None:
        test_df = load_test(output_dir)

    pipe = load_cardiff_pipeline()
    y_pred = predict_cardiff(pipe, test_df["text"].fillna(""))
    y_true = test_df["label"].values

    results = evaluate(y_true, y_pred)
    save_results(results, output_dir)
    print("\n[Step 5] Completed successfully.")
    return results
