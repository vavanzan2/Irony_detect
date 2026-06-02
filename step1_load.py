import os
import pandas as pd
from datasets import load_dataset


def load_epic_dataset():
    print("[Etapa 1] Baixando dataset EPIC do HuggingFace...")
    dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
    print(f"  Splits disponíveis: {list(dataset.keys())}")

    frames = []
    for split_name, split_data in dataset.items():
        df_split = split_data.to_pandas()
        df_split["original_split"] = split_name
        frames.append(df_split)

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total de linhas carregadas: {len(df)}")
    return df


def inspect_columns(df):
    print("\n[Etapa 1] Inspecionando colunas do dataset...")
    print(f"  Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"\n  Colunas disponíveis:\n  {list(df.columns)}")
    print("\n  Tipos de dados:")
    print(df.dtypes.to_string(header=False))
    print("\n  Valores nulos por coluna:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  Nenhum valor nulo encontrado.")
    print("\n  Primeiras 3 linhas:")
    print(df.head(3).to_string())


def inspect_distributions(df):
    print("\n[Etapa 1] Analisando distribuições...")

    label_cols = [c for c in df.columns if "iron" in c.lower() or "label" in c.lower()]
    platform_cols = [c for c in df.columns if "platform" in c.lower() or "source" in c.lower()]
    annotator_cols = [c for c in df.columns if "annot" in c.lower() or "worker" in c.lower()]

    print(f"\n  Colunas de label detectadas:      {label_cols}")
    print(f"  Colunas de plataforma detectadas: {platform_cols}")
    print(f"  Colunas de anotador detectadas:   {annotator_cols}")

    for col in label_cols:
        print(f"\n  Distribuição de '{col}':")
        print(df[col].value_counts(dropna=False).to_string())

    for col in platform_cols:
        print(f"\n  Distribuição de '{col}':")
        print(df[col].value_counts(dropna=False).to_string())

    id_cols = [c for c in df.columns if "id" in c.lower() and "annot" not in c.lower()]
    for col in id_cols[:2]:
        print(f"\n  Valores únicos em '{col}': {df[col].nunique()}")

    for col in annotator_cols[:2]:
        print(f"  Valores únicos em '{col}': {df[col].nunique()}")


def save_raw_data(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "epic_raw.csv")
    df.to_csv(path, index=False)
    print(f"\n[Etapa 1] Dataset bruto salvo em: {path}")
    return path


def run(output_dir="outputs"):
    df = load_epic_dataset()
    inspect_columns(df)
    inspect_distributions(df)
    save_raw_data(df, output_dir)
    print("\n[Etapa 1] Concluída com sucesso.")
    return df
