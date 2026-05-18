import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.utils.paths import ensure_dir
except ModuleNotFoundError:
    from utils.paths import ensure_dir

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def calculate_classification_metrics(
    y_true,
    y_pred,
    average: str = "weighted"
):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return metrics


def save_metrics(
    metrics: dict,
    model_name: str,
    save_dir: str = "outputs/metrics"
):

    ensure_dir(save_dir)

    metrics_with_name = {
        "model": model_name,
        **metrics
    }

    df = pd.DataFrame([metrics_with_name])

    csv_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    json_path = os.path.join(save_dir, f"{model_name}_metrics.json")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_with_name, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Metrics saved to: {csv_path}")
    print(f"[INFO] Metrics saved to: {json_path}")


def save_classification_report(
    y_true,
    y_pred,
    model_name: str,
    labels=None,
    target_names=None,
    save_dir: str = "outputs/metrics"
):

    ensure_dir(save_dir)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    report_path = os.path.join(
        save_dir,
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[INFO] Classification report saved to: {report_path}")
    print("\n" + report)


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    labels=None,
    target_names=None,
    save_dir: str = "outputs/figures"
):
    ensure_dir(save_dir)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    if target_names is not None:
        tick_names = target_names
    elif labels is not None:
        tick_names = labels
    else:
        tick_names = np.unique(y_true)

    tick_marks = np.arange(len(tick_names))
    plt.xticks(tick_marks, tick_names, rotation=45)
    plt.yticks(tick_marks, tick_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    fig_path = os.path.join(
        save_dir,
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Confusion matrix saved to: {fig_path}")


def save_predictions(
    texts,
    y_true,
    y_pred,
    model_name: str,
    probabilities=None,
    save_dir: str = "outputs/predictions"
):

    ensure_dir(save_dir)

    data = {
        "text": texts,
        "true_label": y_true,
        "pred_label": y_pred
    }

    if probabilities is not None:
        data["probability"] = probabilities

    df = pd.DataFrame(data)

    save_path = os.path.join(
        save_dir,
        f"{model_name}_predictions.csv"
    )

    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Predictions saved to: {save_path}")


def save_error_analysis(
    texts,
    y_true,
    y_pred,
    model_name: str,
    probabilities=None,
    save_dir: str = "outputs/errors"
):
    
    ensure_dir(save_dir)

    data = {
        "text": texts,
        "true_label": y_true,
        "pred_label": y_pred
    }

    if probabilities is not None:
        data["probability"] = probabilities

    df = pd.DataFrame(data)

    wrong_df = df[df["true_label"] != df["pred_label"]]

    save_path = os.path.join(
        save_dir,
        f"{model_name}_wrong_cases.csv"
    )

    wrong_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrong cases saved to: {save_path}")
    print(f"[INFO] Number of wrong cases: {len(wrong_df)}")


def evaluate_classification_model(
    y_true,
    y_pred,
    model_name: str,
    texts=None,
    probabilities=None,
    labels=None,
    target_names=None,
    average: str = "weighted",
    save_outputs: bool = True
):

    print("\n" + "=" * 60)
    print(f"Evaluating model: {model_name}")
    print("=" * 60)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        average=average
    )

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    if save_outputs:
        save_metrics(
            metrics=metrics,
            model_name=model_name
        )

        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            labels=labels,
            target_names=target_names
        )

        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            labels=labels,
            target_names=target_names
        )

        if texts is not None:
            save_predictions(
                texts=texts,
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                probabilities=probabilities
            )

            save_error_analysis(
                texts=texts,
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                probabilities=probabilities
            )

    print("=" * 60)
    print(f"Evaluation finished: {model_name}")
    print("=" * 60 + "\n")

    return metrics


def merge_model_metrics(
    model_names,
    metrics_dir: str = "outputs/metrics",
    save_path: str = "outputs/comparison/model_comparison.csv"
):

    all_metrics = []

    for model_name in model_names:
        file_path = os.path.join(metrics_dir, f"{model_name}_metrics.csv")

        if not os.path.exists(file_path):
            print(f"[WARNING] Metrics file not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        all_metrics.append(df)

    if len(all_metrics) == 0:
        print("[WARNING] No metrics files found. Comparison file not created.")
        return None

    comparison_df = pd.concat(all_metrics, axis=0, ignore_index=True)

    ensure_dir(os.path.dirname(save_path))

    comparison_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Model comparison saved to: {save_path}")

    return comparison_df


def plot_model_comparison(
    comparison_csv: str = "outputs/comparison/model_comparison.csv",
    metric: str = "f1",
    save_dir: str = "outputs/comparison"
):

    if not os.path.exists(comparison_csv):
        print(f"[WARNING] Comparison file not found: {comparison_csv}")
        return

    ensure_dir(save_dir)

    df = pd.read_csv(comparison_csv)

    if metric not in df.columns:
        print(f"[WARNING] Metric '{metric}' not found in comparison file.")
        return

    plt.figure(figsize=(7, 5))
    plt.bar(df["model"], df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")

    for i, value in enumerate(df[metric]):
        plt.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"model_comparison_{metric}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Model comparison figure saved to: {save_path}")
