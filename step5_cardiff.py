import json
import os

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
    print(f"[Etapa 5] Teste carregado: {len(df)} textos")
    return df


def load_cardiff_pipeline():
    print(f"[Etapa 5] Baixando modelo Cardiff: {CARDIFF_MODEL}")
    pipe = pipeline(
        "text-classification",
        model=CARDIFF_MODEL,
        tokenizer=CARDIFF_MODEL,
        truncation=True,
        max_length=512,
    )
    print("  Modelo carregado.")
    return pipe


def detect_irony_label(pipe):
    """Descobre qual label string o modelo usa para ironia e retorna o mapeamento label→int."""
    id2label = pipe.model.config.id2label
    print(f"  id2label do Cardiff: {id2label}")

    label_map = {}
    for idx, name in id2label.items():
        n = name.lower().strip()
        if "not" in n or n in ("0", "false"):
            label_map[name] = 0
        else:
            label_map[name] = 1

    print(f"  Mapeamento aplicado: {label_map}")
    return label_map


def predict_cardiff(pipe, texts, batch_size=32):
    """
    Roda o Cardiff em modo zero-shot e retorna predições binárias (1=irônico, 0=não irônico).
    Usa o id2label do modelo para garantir o mapeamento correto.
    """
    label_map = detect_irony_label(pipe)

    print(f"[Etapa 5] Predizendo {len(texts)} textos em batches de {batch_size}...")
    outputs = pipe(list(texts), batch_size=batch_size, truncation=True, max_length=512)

    # Diagnóstico: mostra as primeiras 3 predições brutas
    print(f"  Exemplos de saída bruta do pipeline: {outputs[:3]}")

    preds = []
    for out in outputs:
        raw_label = out["label"]
        # Usa o mapeamento detectado; fallback por string se não encontrar
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
    report = classification_report(y_true, y_pred, target_names=["não irônico", "irônico"])

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

    print("\n[Etapa 5] Resultados do Cardiff (zero-shot):")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Macro-F1  : {results['macro_f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print("\n  Matriz de Confusão (linhas=real, colunas=previsto):")
    print("              Irônico  Não Irônico")
    print(f"  Irônico    {cm[0][0]:>7}  {cm[0][1]:>11}")
    print(f"  Não Irônico{cm[1][0]:>7}  {cm[1][1]:>11}")
    print(f"\n{report}")
    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cardiff_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Etapa 5] Resultados salvos em: {path}")


def run(test_df=None, output_dir="outputs"):
    if test_df is None:
        test_df = load_test(output_dir)

    pipe = load_cardiff_pipeline()
    y_pred = predict_cardiff(pipe, test_df["text"].fillna(""))
    y_true = test_df["label"].values

    results = evaluate(y_true, y_pred)
    save_results(results, output_dir)
    print("\n[Etapa 5] Concluída com sucesso.")
    return results
