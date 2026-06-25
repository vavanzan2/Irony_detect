import csv
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
            print(f"  Loaded: {path}")
        else:
            print(f"  WARNING: not found — {path}")
    return results


def print_comparison_table(results):
    if not results:
        print("No results available to compare.")
        return

    col_w = 28
    met_w = 12
    header = f"{'Model':<{col_w}}" + "".join(f"{m.upper():>{met_w}}" for m in METRICS)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  COMPARISON TABLE — IRONY DETECTION (EPIC)")
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

    # Best model by Macro-F1
    best_name = max(results, key=lambda n: results[n].get("macro_f1", 0))
    print(f"\n  Best Macro-F1: {best_name} ({results[best_name].get('macro_f1', 0):.4f})")


def print_confusion_matrices(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("  CONFUSION MATRICES")
    print("  (rows = actual, cols = predicted)")
    print("  Order: [Ironic, Not Ironic]")
    print("=" * 60)

    for name, res in results.items():
        cm = res.get("confusion_matrix")
        if not cm:
            continue
        print(f"\n  {name}")
        print(f"  {'':20} Predicted: Ironic  Predicted: Not Ironic")
        print(f"  {'Actual: Ironic':20} {cm[0][0]:>16}  {cm[0][1]:>21}")
        print(f"  {'Actual: Not Ironic':20} {cm[1][0]:>16}  {cm[1][1]:>21}")


def analyze_errors(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS")
    print("=" * 60)

    for name, res in results.items():
        cm = res.get("confusion_matrix")
        if not cm:
            continue
        tp = cm[0][0]  # ironic predicted as ironic
        fn = cm[0][1]  # ironic predicted as not ironic  (false negative)
        fp = cm[1][0]  # not ironic predicted as ironic  (false positive)
        tn = cm[1][1]  # not ironic predicted as not ironic

        total = tp + fn + fp + tn
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\n  {name}")
        print(f"    TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"    False Positive Rate (sees irony too often): {fp_rate:.2%}")
        print(f"    False Negative Rate (misses real irony): {fn_rate:.2%}")

        if fp_rate > fn_rate:
            print("    → Tendency: 'aggressive' model — prefers to predict irony when unsure")
        else:
            print("    → Tendency: 'conservative' model — prefers to predict not irony when unsure")

    # Difference in behavior between models
    if len(results) >= 2:
        print("\n  ERROR PATTERN COMPARISON:")
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
                    print(f"  {n1} vs {n2}: difference in FP rate = {diff:.2%}")


def save_csv(results, output_dir):
    if not results:
        return

    metrics_path = os.path.join(output_dir, "comparison_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + METRICS)
        writer.writeheader()
        for name, res in results.items():
            row = {"model": name}
            for m in METRICS:
                row[m] = res.get(m, "N/A")
            writer.writerow(row)
    print(f"[Step 8] Metrics saved at: {metrics_path}")

    cm_path = os.path.join(output_dir, "comparison_confusion_matrices.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "TP", "FN", "FP", "TN"])
        writer.writeheader()
        for name, res in results.items():
            cm = res.get("confusion_matrix")
            if not cm:
                continue
            writer.writerow({
                "model": name,
                "TP": cm[0][0],
                "FN": cm[0][1],
                "FP": cm[1][0],
                "TN": cm[1][1],
            })
    print(f"[Step 8] Confusion matrices saved at: {cm_path}")


def run(output_dir="outputs"):
    print("[Step 8] Loading saved results...")
    results = load_results(output_dir)

    print_comparison_table(results)
    print_confusion_matrices(results)
    analyze_errors(results)
    save_csv(results, output_dir)

    print("\n[Step 8] Completed successfully.")
    return results
