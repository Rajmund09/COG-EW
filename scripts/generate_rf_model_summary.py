"""
COG-EW RF MODEL PERFORMANCE SUMMARY

Reads the previously generated evaluation JSON and
produces a concise model-performance summary.

Input:
    reports/cogew_rf_cnn_test_evaluation.json

Output:
    reports/cogew_rf_model_summary.json
    reports/cogew_rf_model_summary.txt
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "reports" / "cogew_rf_cnn_test_evaluation.json"

OUTPUT_JSON = PROJECT_ROOT / "reports" / "cogew_rf_model_summary.json"

OUTPUT_TXT = PROJECT_ROOT / "reports" / "cogew_rf_model_summary.txt"


def main():

    print("=" * 70)
    print("COG-EW RF MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(f"Evaluation file not found:\n" f"{INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Extract available fields robustly
    # --------------------------------------------------------

    accuracy = data.get("test_accuracy", data.get("accuracy"))

    class_metrics = data.get("classification_metrics", data.get("class_metrics", {}))

    snr_metrics = data.get("accuracy_by_snr", data.get("snr_accuracy", {}))

    source_metrics = data.get("accuracy_by_source", data.get("source_accuracy", {}))

    # --------------------------------------------------------
    # Per-class analysis
    # --------------------------------------------------------

    class_summary = {}

    for class_name, metrics in class_metrics.items():

        class_summary[class_name] = {
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score", metrics.get("f1")),
            "support": metrics.get("support", metrics.get("samples")),
        }

    # --------------------------------------------------------
    # Identify strongest / weakest classes
    # --------------------------------------------------------

    valid_classes = [
        (name, values["f1_score"])
        for name, values in class_summary.items()
        if values["f1_score"] is not None
    ]

    strongest = None
    weakest = None

    if valid_classes:

        strongest = max(valid_classes, key=lambda x: x[1])

        weakest = min(valid_classes, key=lambda x: x[1])

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = {
        "model": ("COG-EW RF Automatic " "Modulation Classification CNN"),
        "test_accuracy": accuracy,
        "classes": class_summary,
        "strongest_class": (
            {"class": strongest[0], "f1_score": strongest[1]} if strongest else None
        ),
        "weakest_class": (
            {"class": weakest[0], "f1_score": weakest[1]} if weakest else None
        ),
        "accuracy_by_snr": snr_metrics,
        "accuracy_by_source": source_metrics,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2)

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    lines = []

    lines.append("COG-EW RF MODEL PERFORMANCE SUMMARY")

    lines.append("=" * 70)

    lines.append("")

    if accuracy is not None:

        lines.append(f"Test Accuracy : " f"{float(accuracy) * 100:.2f}%")

    lines.append("")

    lines.append("CLASS PERFORMANCE")
    lines.append("-" * 70)

    for class_name, metrics in class_summary.items():

        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1_score"]
        support = metrics["support"]

        def pct(value):

            if value is None:
                return "N/A"

            return f"{float(value) * 100:.2f}%"

        lines.append(
            f"{class_name:<10} "
            f"Precision={pct(precision):>8} "
            f"Recall={pct(recall):>8} "
            f"F1={pct(f1):>8} "
            f"Samples={support}"
        )

    lines.append("")

    if strongest:

        lines.append(
            f"Strongest class: " f"{strongest[0]} " f"(F1={strongest[1] * 100:.2f}%)"
        )

    if weakest:

        lines.append(
            f"Weakest class: " f"{weakest[0]} " f"(F1={weakest[1] * 100:.2f}%)"
        )

    lines.append("")

    lines.append("SNR PERFORMANCE")

    lines.append("-" * 70)

    if isinstance(snr_metrics, dict):

        for snr, value in snr_metrics.items():

            if isinstance(value, dict):

                acc = value.get("accuracy", value.get("acc"))

            else:

                acc = value

            if acc is not None:

                lines.append(f"{snr:>8} : " f"{float(acc) * 100:.2f}%")

    lines.append("")

    lines.append("DATASET SOURCE PERFORMANCE")

    lines.append("-" * 70)

    if isinstance(source_metrics, dict):

        for source, value in source_metrics.items():

            if isinstance(value, dict):

                acc = value.get("accuracy", value.get("acc"))

            else:

                acc = value

            if acc is not None:

                lines.append(f"{source:<15} " f"{float(acc) * 100:.2f}%")

    report_text = "\n".join(lines)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:

        f.write(report_text)

    print(report_text)

    print()
    print("=" * 70)

    print(f"[OK] JSON summary:\n" f"{OUTPUT_JSON}")

    print(f"[OK] Text summary:\n" f"{OUTPUT_TXT}")

    print("=" * 70)


if __name__ == "__main__":

    main()
