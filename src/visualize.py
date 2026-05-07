import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MODELS = {
    "lr_tfidf": "TF-IDF + LR",
    "lstm": "LSTM",
    "bert": "BERT",
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_series(history, key):
    values = history.get(key, [])
    return values if isinstance(values, list) else []


def last_value(values):
    return values[-1] if values else None


def best_epoch(values):
    if not values:
        return None
    return max(range(len(values)), key=values.__getitem__) + 1


def round_metric(value):
    if value is None:
        return None
    return round(float(value), 6)


def build_model_summary(model_name, parameters_dir, display_name=None):
    model_dir = Path(parameters_dir) / model_name
    metrics = load_json(model_dir / "metrics.json")
    history_payload = load_json(model_dir / "history.json") or {}
    history = history_payload.get("history", {})

    if metrics is None:
        print(f"[WARNING] Metrics file not found: {model_dir / 'metrics.json'}")
        return None, {}

    train_acc = get_series(history, "train_acc")
    val_acc = get_series(history, "val_acc")
    train_loss = get_series(history, "train_loss")
    val_loss = get_series(history, "val_loss")
    train_f1 = get_series(history, "train_f1")
    val_f1 = get_series(history, "val_f1")

    final_train_acc = last_value(train_acc)
    final_val_acc = last_value(val_acc)
    final_train_f1 = last_value(train_f1)
    final_val_f1 = last_value(val_f1)

    row = {
        "model": display_name or model_name,
        "model_name": model_name,
        "test_accuracy": round_metric(metrics.get("test_accuracy")),
        "test_macro_f1": round_metric(metrics.get("test_macro_f1")),
        "test_loss": round_metric(metrics.get("test_loss")),
        "best_val_f1": round_metric(metrics.get("best_val_f1")),
        "best_cv_score": round_metric(metrics.get("best_cv_score")),
        "error_rate": round_metric(
            1 - metrics["test_accuracy"] if metrics.get("test_accuracy") is not None else None
        ),
        "epochs": max(
            len(train_acc),
            len(val_acc),
            len(train_loss),
            len(val_loss),
            len(train_f1),
            len(val_f1),
            0,
        ),
        "best_epoch_by_val_f1": best_epoch(val_f1),
        "final_train_accuracy": round_metric(final_train_acc),
        "final_val_accuracy": round_metric(final_val_acc),
        "final_train_macro_f1": round_metric(final_train_f1),
        "final_val_macro_f1": round_metric(final_val_f1),
        "final_train_loss": round_metric(last_value(train_loss)),
        "final_val_loss": round_metric(last_value(val_loss)),
        "train_val_accuracy_gap": round_metric(
            final_train_acc - final_val_acc
            if final_train_acc is not None and final_val_acc is not None
            else None
        ),
        "train_val_macro_f1_gap": round_metric(
            final_train_f1 - final_val_f1
            if final_train_f1 is not None and final_val_f1 is not None
            else None
        ),
        "metrics_saved_at": metrics.get("saved_at"),
        "history_saved_at": history_payload.get("saved_at"),
    }

    return row, history


def save_summary(summary_df, output_dir):
    csv_path = Path(output_dir) / "model_metrics_summary.csv"
    json_path = Path(output_dir) / "model_metrics_summary.json"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_json(json_path, orient="records", force_ascii=False, indent=4)

    print(f"[INFO] Metrics summary saved to: {csv_path}")
    print(f"[INFO] Metrics summary saved to: {json_path}")


def configure_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def annotate_bars(ax, bars, padding=0.005):
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + padding,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_test_scores(summary_df, output_dir):
    metric_names = ["test_accuracy", "test_macro_f1"]
    metric_labels = ["Test Accuracy", "Test Macro F1"]
    available = [
        (name, label)
        for name, label in zip(metric_names, metric_labels)
        if name in summary_df.columns and summary_df[name].notna().any()
    ]

    if not available:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = list(range(len(summary_df)))
    width = 0.35

    for index, (metric, label) in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            summary_df[metric],
            width=width,
            label=label,
        )
        annotate_bars(ax, bars)

    ax.set_title("Model Test Scores")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_df["model"], rotation=0)
    ax.set_ylim(0, min(1.08, max(summary_df[[name for name, _ in available]].max()) + 0.08))
    ax.legend()
    fig.tight_layout()

    save_path = Path(output_dir) / "model_test_scores.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Test score figure saved to: {save_path}")


def plot_metric_heatmap(summary_df, output_dir):
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "best_val_f1",
        "best_cv_score",
        "error_rate",
    ]
    metrics = [
        metric
        for metric in metrics
        if metric in summary_df.columns and summary_df[metric].notna().any()
    ]

    if not metrics:
        return

    matrix = summary_df[metrics].astype(float)

    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 3, 0.65 * len(summary_df) + 2.2))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Model Metrics Heatmap")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(summary_df)))
    ax.set_yticklabels(summary_df["model"])

    for row_index in range(len(summary_df)):
        for col_index, metric in enumerate(metrics):
            value = matrix.iloc[row_index, col_index]
            text = "" if pd.isna(value) else f"{value:.4f}"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()

    save_path = Path(output_dir) / "model_metric_heatmap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Metrics heatmap saved to: {save_path}")


def plot_test_loss(summary_df, output_dir):
    if "test_loss" not in summary_df.columns or not summary_df["test_loss"].notna().any():
        return

    loss_df = summary_df[summary_df["test_loss"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(loss_df["model"], loss_df["test_loss"], color="#d95f02")
    annotate_bars(ax, bars, padding=max(loss_df["test_loss"].max() * 0.02, 0.002))
    ax.set_title("Model Test Loss")
    ax.set_xlabel("Model")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, loss_df["test_loss"].max() * 1.18)
    fig.tight_layout()

    save_path = Path(output_dir) / "model_test_loss.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Test loss figure saved to: {save_path}")


def plot_training_curve(histories, metric, ylabel, output_dir):
    has_values = any(
        get_series(history, f"train_{metric}") or get_series(history, f"val_{metric}")
        for history in histories.values()
    )
    if not has_values:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, history in histories.items():
        display_name = DEFAULT_MODELS.get(model_name, model_name)
        for split, line_style in [("train", "-"), ("val", "--")]:
            values = get_series(history, f"{split}_{metric}")
            if not values:
                continue
            epochs = list(range(1, len(values) + 1))
            ax.plot(
                epochs,
                values,
                linestyle=line_style,
                marker="o",
                linewidth=2,
                label=f"{display_name} {split}",
            )

    ax.set_title(f"Training {ylabel} Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    epoch_ticks = sorted(
        {
            epoch
            for history in histories.values()
            for split in ["train", "val"]
            for epoch in range(1, len(get_series(history, f"{split}_{metric}")) + 1)
        }
    )
    if epoch_ticks:
        ax.set_xticks(epoch_ticks)
    if metric in {"acc", "f1"}:
        ax.set_ylim(0, 1.05)
    ax.legend(ncol=2)
    fig.tight_layout()

    save_path = Path(output_dir) / f"training_{metric}_curves.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Training {metric} curve saved to: {save_path}")


def plot_generalization_gap(summary_df, output_dir):
    gap_columns = ["train_val_accuracy_gap", "train_val_macro_f1_gap"]
    if not any(column in summary_df.columns and summary_df[column].notna().any() for column in gap_columns):
        return

    gap_df = summary_df[summary_df[gap_columns].notna().any(axis=1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = list(range(len(gap_df)))
    width = 0.35

    labels = {
        "train_val_accuracy_gap": "Accuracy Gap",
        "train_val_macro_f1_gap": "Macro F1 Gap",
    }

    for index, column in enumerate(gap_columns):
        offset = (index - 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            gap_df[column],
            width=width,
            label=labels[column],
        )
        annotate_bars(ax, bars, padding=0.003)

    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Train-Val Generalization Gap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Final Train - Final Val")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(gap_df["model"])
    ax.legend()
    fig.tight_layout()

    save_path = Path(output_dir) / "model_generalization_gap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Generalization gap figure saved to: {save_path}")


def visualize(models, parameters_dir, output_dir):
    parameters_dir = Path(parameters_dir)
    output_dir = ensure_dir(output_dir)

    rows = []
    histories = {}

    for model_name in models:
        row, history = build_model_summary(
            model_name=model_name,
            parameters_dir=parameters_dir,
            display_name=DEFAULT_MODELS.get(model_name, model_name),
        )
        if row is None:
            continue
        rows.append(row)
        if history:
            histories[model_name] = history

    if not rows:
        raise FileNotFoundError(
            f"No model metrics were found under: {parameters_dir}"
        )

    summary_df = pd.DataFrame(rows)
    save_summary(summary_df, output_dir)

    configure_plot_style()
    plot_test_scores(summary_df, output_dir)
    plot_metric_heatmap(summary_df, output_dir)
    plot_test_loss(summary_df, output_dir)
    plot_generalization_gap(summary_df, output_dir)
    plot_training_curve(histories, metric="loss", ylabel="Loss", output_dir=output_dir)
    plot_training_curve(histories, metric="acc", ylabel="Accuracy", output_dir=output_dir)
    plot_training_curve(histories, metric="f1", ylabel="Macro F1", output_dir=output_dir)

    print("\nModel metrics summary:")
    print(summary_df.to_string(index=False))
    return summary_df


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Compare model training results and save visualizations."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS.keys()),
        help="Model directory names under parameters/.",
    )
    parser.add_argument(
        "--parameters-dir",
        default=str(PROJECT_ROOT / "parameters"),
        help="Directory that stores each model's metrics.json/history.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory to save summary files and figures.",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed_args = parse_args(args)
    return visualize(
        models=parsed_args.models,
        parameters_dir=parsed_args.parameters_dir,
        output_dir=parsed_args.output_dir,
    )


if __name__ == "__main__":
    main()
