import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_splits(output_dir):
    train_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(output_dir, "test.csv"))
    print(f"[Etapa 4] Treino: {len(train_df)} | Teste: {len(test_df)}")
    return train_df, test_df


def train_baseline(train_df):
    """Treina TF-IDF + Regressão Logística no conjunto de treino."""
    print("[Etapa 4] Treinando TF-IDF + Logistic Regression...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    y_train = train_df["label"].values

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    print("  Modelo treinado.")
    return vectorizer, clf


def evaluate(vectorizer, clf, test_df):
    """Avalia o modelo no conjunto de teste e retorna dicionário de métricas."""
    X_test = vectorizer.transform(test_df["text"].fillna(""))
    y_true = test_df["label"].values
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    report = classification_report(y_true, y_pred, target_names=["não irônico", "irônico"])

    results = {
        "model":     "TF-IDF + Logistic Regression",
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("\n[Etapa 4] Resultados do Baseline:")
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
    path = os.path.join(output_dir, "baseline_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Etapa 4] Resultados salvos em: {path}")


def run(train_df=None, test_df=None, output_dir="outputs"):
    if train_df is None or test_df is None:
        train_df, test_df = load_splits(output_dir)

    vectorizer, clf = train_baseline(train_df)
    results = evaluate(vectorizer, clf, test_df)
    save_results(results, output_dir)
    print("\n[Etapa 4] Concluída com sucesso.")
    return results
