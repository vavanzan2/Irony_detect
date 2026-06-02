import json
import os


RESULT_FILES = {
    "Baseline (TF-IDF + LR)":    "baseline_results.json",
    "Cardiff (zero-shot)":        "cardiff_results.json",
    "RoBERTa fine-tuned (EPIC)": "finetuned_results.json",
}

METRICS = ["accuracy", "precision", "recall", "macro_f1"]


def load_results(output_dir):
    results = {}
    for name, filename in RESULT_FILES.items():
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                results[name] = json.load(f)
            print(f"  Carregado: {path}")
        else:
            print(f"  AVISO: não encontrado — {path}")
    return results


def print_comparison_table(results):
    if not results:
        print("Nenhum resultado disponível para comparar.")
        return

    col_w = 28
    met_w = 12
    header = f"{'Modelo':<{col_w}}" + "".join(f"{m.upper():>{met_w}}" for m in METRICS)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  TABELA COMPARATIVA — DETECÇÃO DE IRONIA (EPIC)")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, res in results.items():
        row = f"{name:<{col_w}}"
        for m in METRICS:
            val = res.get(m, "N/A")
            row += f"{val:>{met_w}.4f}" if isinstance(val, float) else f"{'N/A':>{met_w}}"
        print(row)

    print(sep)

    # Melhor modelo por Macro-F1
    best_name = max(results, key=lambda n: results[n].get("macro_f1", 0))
    print(f"\n  Melhor Macro-F1: {best_name} ({results[best_name].get('macro_f1', 0):.4f})")


def print_confusion_matrices(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("  MATRIZES DE CONFUSÃO")
    print("  (linhas = real, colunas = previsto)")
    print("  Ordem: [Irônico, Não Irônico]")
    print("=" * 60)

    for name, res in results.items():
        cm = res.get("confusion_matrix")
        if not cm:
            continue
        print(f"\n  {name}")
        print(f"  {'':20} Previsto: Irônico  Previsto: Não Irônico")
        print(f"  {'Real: Irônico':20} {cm[0][0]:>16}  {cm[0][1]:>21}")
        print(f"  {'Real: Não Irônico':20} {cm[1][0]:>16}  {cm[1][1]:>21}")


def analyze_errors(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("  ANÁLISE DE ERROS")
    print("=" * 60)

    for name, res in results.items():
        cm = res.get("confusion_matrix")
        if not cm:
            continue
        tp = cm[0][0]  # irônico previsto como irônico
        fn = cm[0][1]  # irônico previsto como não irônico  (falso negativo)
        fp = cm[1][0]  # não irônico previsto como irônico  (falso positivo)
        tn = cm[1][1]  # não irônico previsto como não irônico

        total = tp + fn + fp + tn
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\n  {name}")
        print(f"    TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"    Taxa de Falsos Positivos (vê ironia demais): {fp_rate:.2%}")
        print(f"    Taxa de Falsos Negativos (perde ironia real): {fn_rate:.2%}")

        if fp_rate > fn_rate:
            print("    → Tendência: modelo 'agressivo' — prefere prever ironia quando incerto")
        else:
            print("    → Tendência: modelo 'conservador' — prefere prever não ironia quando incerto")

    # Diferença de comportamento entre modelos
    if len(results) >= 2:
        print("\n  COMPARAÇÃO DE PADRÕES DE ERRO:")
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                cm1 = results[n1].get("confusion_matrix")
                cm2 = results[n2].get("confusion_matrix")
                if cm1 and cm2:
                    fp1_rate = cm1[1][0] / max(1, cm1[1][0] + cm1[1][1])
                    fp2_rate = cm2[1][0] / max(1, cm2[1][0] + cm2[1][1])
                    diff = abs(fp1_rate - fp2_rate)
                    print(f"  {n1} vs {n2}: diferença de FP rate = {diff:.2%}")


def run(output_dir="outputs"):
    print("[Etapa 8] Carregando resultados salvos...")
    results = load_results(output_dir)

    print_comparison_table(results)
    print_confusion_matrices(results)
    analyze_errors(results)

    print("\n[Etapa 8] Concluída com sucesso.")
    return results
