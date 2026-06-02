import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset


BASE_MODEL = "FacebookAI/roberta-base"


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class IronyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def load_splits(output_dir):
    train_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(output_dir, "val.csv"))
    print(f"[Etapa 6] Treino: {len(train_df)} | Validação: {len(val_df)}")
    return train_df, val_df


def tokenize(tokenizer, texts, max_length=128):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy  = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": accuracy}


def finetune(train_df, val_df, output_dir, epochs=3, batch_size=16, lr=2e-5):
    model_save_path = os.path.join(output_dir, "roberta_finetuned")

    print(f"[Etapa 6] Carregando tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("[Etapa 6] Tokenizando dados...")
    train_enc = tokenize(tokenizer, train_df["text"].fillna(""))
    val_enc   = tokenize(tokenizer, val_df["text"].fillna(""))

    train_dataset = IronyDataset(train_enc, train_df["label"].tolist())
    val_dataset   = IronyDataset(val_enc,   val_df["label"].tolist())

    print(f"[Etapa 6] Carregando modelo base: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "not irony", 1: "irony"},
        label2id={"not irony": 0, "irony": 1},
    )

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"[Etapa 6] Iniciando fine-tuning — {epochs} épocas, lr={lr}, batch={batch_size}")
    trainer.train()

    print(f"[Etapa 6] Salvando modelo em: {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("[Etapa 6] Modelo salvo com sucesso.")
    return model_save_path


def run(train_df=None, val_df=None, output_dir="outputs", epochs=3, batch_size=16, lr=2e-5):
    if train_df is None or val_df is None:
        train_df, val_df = load_splits(output_dir)

    model_path = finetune(train_df, val_df, output_dir, epochs=epochs, batch_size=batch_size, lr=lr)
    print("\n[Etapa 6] Concluída com sucesso.")
    return model_path
