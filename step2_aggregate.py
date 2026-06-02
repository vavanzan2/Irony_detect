import os
import pandas as pd


def detect_columns(df):
    """Detecta os nomes das colunas relevantes no DataFrame bruto."""
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
    print(f"  Colunas detectadas: {detected}")
    return detected


def encode_label(value):
    """Converte o valor de ironia para int (1=irônico, 0=não irônico).
    Cobre os valores reais do EPIC: 'iro' / 'not'.
    """
    if isinstance(value, (int, float)):
        return int(bool(value))
    v = str(value).strip().lower()
    if v in ("iro", "irony", "ironic", "1", "true", "yes"):
        return 1
    return 0


def majority_vote(df, cols):
    """Agrega as anotações por item_id via voto majoritário.
    Retorna um DataFrame com uma linha por texto.
    """
    print("[Etapa 2] Aplicando voto majoritário...")

    id_col    = cols["item_id"]
    label_col = cols["label"]
    reply_col = cols["reply"]
    post_col  = cols["post"]
    plat_col  = cols["platform"]

    df = df.copy()
    df["_label_int"] = df[label_col].apply(encode_label)

    # Contagem de votos por texto (named aggregation — sem multi-index)
    votes = (
        df.groupby(id_col)["_label_int"]
        .agg(vote_ironic="sum", n_annotators="count")
        .reset_index()
    )
    votes.rename(columns={id_col: "item_id"}, inplace=True)
    votes["label"] = (votes["vote_ironic"] >= votes["n_annotators"] / 2).astype(int)

    # Primeiro valor de texto/plataforma por texto
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

    print(f"  Textos únicos: {len(labeled)}")
    print(f"  Distribuição final:\n{labeled['label'].value_counts().to_string()}")
    print(f"  (1 = irônico, 0 = não irônico)")
    return labeled


def save_labeled(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "epic_labeled.csv")
    df.to_csv(path, index=False)
    print(f"\n[Etapa 2] Dataset com labels salvo em: {path}")
    return path


def run(raw_df=None, output_dir="outputs"):
    if raw_df is None:
        path = os.path.join(output_dir, "epic_raw.csv")
        print(f"[Etapa 2] Carregando {path}...")
        raw_df = pd.read_csv(path)

    cols = detect_columns(raw_df)
    labeled_df = majority_vote(raw_df, cols)
    save_labeled(labeled_df, output_dir)
    print("\n[Etapa 2] Concluída com sucesso.")
    return labeled_df
