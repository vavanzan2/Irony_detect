import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class IronyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def load_model(model_dir):
    print(f"[Step 7] Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print("  Model loaded.")
    return tokenizer, model


def load_test(output_dir):
    path = os.path.join(output_dir, "test.csv")
    df = pd.read_csv(path)
    print(f"[Step 7] Test set: {len(df)} texts")
    return df


def predict(tokenizer, model, texts, batch_size=32, max_length=128, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[Step 7] Inference device: {device}")
    print(f"[Step 7] Classification threshold: {threshold}")

    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = IronyDataset({k: v.tolist() for k, v in encodings.items()})
    loader  = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = (probs[:, 1] >= threshold).astype(int)
            all_preds.extend(preds.tolist())

    return all_preds


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    report = classification_report(y_true, y_pred, target_names=["not ironic", "ironic"])

    results = {
        "model":     "RoBERTa fine-tuned (EPIC)",
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("\n[Step 7] RoBERTa fine-tuned results:")
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
    path = os.path.join(output_dir, "finetuned_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Step 7] Results saved at: {path}")


def run(test_df=None, output_dir="outputs", threshold=0.3):
    model_dir = os.path.join(output_dir, "roberta_finetuned")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found at '{model_dir}'. Run the 'finetune' step first."
        )

    if test_df is None:
        test_df = load_test(output_dir)

    tokenizer, model = load_model(model_dir)
    y_pred = predict(tokenizer, model, test_df["text"].fillna(""), threshold=threshold)
    y_true = test_df["label"].values

    results = evaluate(y_true, y_pred)
    results["threshold"] = threshold
    save_results(results, output_dir)
    print("\n[Step 7] Completed successfully.")
    return results
