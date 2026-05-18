import argparse
import json
import os
from pathlib import Path


try:
    from src.utils.paths import COMPARISON_DIR, CONFIGS_DIR, FGM_COMPARISON_DIR, OUTPUTS_DIR, ensure_dir
except ModuleNotFoundError:
    from utils.paths import COMPARISON_DIR, CONFIGS_DIR, FGM_COMPARISON_DIR, OUTPUTS_DIR, ensure_dir


MPL_CONFIG_DIR = OUTPUTS_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MODELS = {
    "lr_tfidf": "TF-IDF + LR",
    "lstm": "LSTM",
    "bert_partial_last_8_no_fgm": "BERT",
}

FGM_COMPARISON_LAYERS = [4, 8]

def load_json(path):
    path = Path(path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8-sig") as f:
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


def freeze_row_uses_fgm(row):
    value = row.get("use_fgm", False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _first_existing(row, keys, default=None):
    for key in keys:
        if key in row and not pd.isna(row[key]):
            return row[key]
    return default


def _metric_value(row, *keys):
    value = _first_existing(row, keys)
    return None if value is None else float(value)


def freeze_row_embedding_unfrozen(row):
    value = row.get("embedding_unfrozen", False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def freeze_strategy_label(row):
    if row["finetune_strategy"] == "partial":
        base_label = f"last {int(row['unfreeze_last_n_layers'])}"
    else:
        base_label = str(row["finetune_strategy"])

    if freeze_row_uses_fgm(row) and freeze_row_embedding_unfrozen(row):
        fgm_label = "embedding_fgm"
    elif freeze_row_uses_fgm(row):
        fgm_label = "fgm"
    else:
        fgm_label = "no_fgm"
    return f"{base_label} {fgm_label}"


def prepare_bert_freeze_summary(summary_df):
    summary_df = summary_df.copy()
    summary_df["strategy_label"] = summary_df.apply(freeze_strategy_label, axis=1)
    return summary_df


def build_bert_freeze_summary_from_configs(configs_dir=CONFIGS_DIR):
    configs_dir = Path(configs_dir)
    rows = []

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    for config_path in sorted(path for path in configs_dir.iterdir() if path.is_dir()):
        config = load_json(config_path / "config.json")
        metrics = load_json(config_path / "metrics.json")
        history_payload = load_json(config_path / "history.json") or {}
        history = history_payload.get("history", {})

        if not config or not metrics:
            continue
        if not str(config.get("model_name", config_path.name)).startswith("bert"):
            continue

        train_acc = get_series(history, "train_acc")
        train_f1 = get_series(history, "train_f1")
        val_acc = get_series(history, "val_acc")
        val_f1 = get_series(history, "val_f1")
        val_loss = get_series(history, "val_loss")
        best_index = None
        if val_f1:
            best_index = max(range(len(val_f1)), key=val_f1.__getitem__)

        trainable_ratio = _metric_value(metrics, "trainable_ratio")
        if trainable_ratio is None:
            trainable_ratio = _metric_value(config, "trainable_ratio")

        rows.append(
            {
                "experiment_name": config.get("model_name", config_path.name),
                "run_name": config_path.name,
                "finetune_strategy": config.get("finetune_strategy", "full"),
                "unfreeze_last_n_layers": config.get("unfreeze_last_n_layers"),
                "use_fgm": config.get("use_fgm", False),
                "fgm_epsilon": config.get("fgm_epsilon", 1.0),
                "embedding_unfrozen": config.get("embedding_unfrozen", False),
                "trainable_ratio": trainable_ratio,
                "trainable_ratio_percent": None if trainable_ratio is None else trainable_ratio * 100,
                "trainable_params": _metric_value(metrics, "trainable_params") or _metric_value(config, "trainable_params"),
                "total_params": _metric_value(metrics, "total_params") or _metric_value(config, "total_params"),
                "train_acc": _metric_value(metrics, "train_acc", "train_accuracy") or last_value(train_acc),
                "train_f1": _metric_value(metrics, "train_f1", "train_macro_f1") or last_value(train_f1),
                "test_acc": _metric_value(metrics, "test_acc", "test_accuracy"),
                "test_f1": _metric_value(metrics, "test_f1", "test_macro_f1"),
                "test_loss": _metric_value(metrics, "test_loss"),
                "best_epoch_by_val_f1": None if best_index is None else best_index + 1,
                "best_val_loss": None if best_index is None or not val_loss else val_loss[best_index],
                "best_val_acc": None if best_index is None or not val_acc else val_acc[best_index],
                "best_val_f1": _metric_value(metrics, "best_val_f1") or (None if best_index is None else val_f1[best_index]),
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "max_len": config.get("max_len"),
                "bert_lr": config.get("bert_lr"),
                "classifier_lr": config.get("classifier_lr"),
                "weight_decay": config.get("weight_decay"),
                "metrics_saved_at": metrics.get("saved_at"),
                "history_saved_at": history_payload.get("saved_at"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return prepare_bert_freeze_summary(pd.DataFrame(rows))


def plot_bert_freeze_scores(summary_df, output_dir):
    metrics = [
        ("train_acc", "Train Acc"),
        ("train_f1", "Train F1"),
        ("test_acc", "Test Acc"),
        ("test_f1", "Test F1"),
    ]
    metrics = [
        (metric, label)
        for metric, label in metrics
        if metric in summary_df.columns and summary_df[metric].notna().any()
    ]

    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(summary_df)), 5.5))
    x_positions = list(range(len(summary_df)))
    width = min(0.18, 0.75 / len(metrics))

    for index, (metric, label) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            summary_df[metric],
            width=width,
            label=label,
        )
        annotate_bars(ax, bars, padding=0.004)

    ax.set_title("BERT Freeze Strategy Scores")
    ax.set_xlabel("Freeze Strategy")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_df["strategy_label"], rotation=25, ha="right")
    ax.set_ylim(0, min(1.08, max(summary_df[[metric for metric, _ in metrics]].max()) + 0.08))
    ax.legend(ncol=2)
    fig.tight_layout()

    save_path = Path(output_dir) / "bert_freeze_scores.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] BERT freeze score figure saved to: {save_path}")


def plot_bert_freeze_trainable(summary_df, output_dir):
    required = {"trainable_ratio_percent", "trainable_params"}
    if not required <= set(summary_df.columns):
        return

    fig, ax_ratio = plt.subplots(figsize=(max(9, 1.2 * len(summary_df)), 5))
    bars = ax_ratio.bar(
        summary_df["strategy_label"],
        summary_df["trainable_ratio_percent"],
        color="#4c78a8",
        label="Trainable Ratio (%)",
    )
    annotate_bars(ax_ratio, bars, padding=max(summary_df["trainable_ratio_percent"].max() * 0.02, 0.1))
    ax_ratio.set_title("BERT Trainable Parameters by Freeze Strategy")
    ax_ratio.set_xlabel("Freeze Strategy")
    ax_ratio.set_ylabel("Trainable Ratio (%)")
    ax_ratio.tick_params(axis="x", rotation=25)

    ax_params = ax_ratio.twinx()
    params_millions = summary_df["trainable_params"] / 1_000_000
    ax_params.plot(
        summary_df["strategy_label"],
        params_millions,
        color="#d95f02",
        marker="o",
        linewidth=2,
        label="Trainable Params (M)",
    )
    ax_params.set_ylabel("Trainable Params (M)")

    lines, labels = ax_ratio.get_legend_handles_labels()
    lines_2, labels_2 = ax_params.get_legend_handles_labels()
    ax_ratio.legend(lines + lines_2, labels + labels_2, loc="upper left")
    fig.tight_layout()

    save_path = Path(output_dir) / "bert_freeze_trainable_params.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] BERT freeze trainable parameter figure saved to: {save_path}")


def plot_bert_freeze_generalization_gap(summary_df, output_dir):
    if not {"train_acc", "test_acc", "train_f1", "test_f1"} <= set(summary_df.columns):
        return

    gap_df = summary_df.copy()
    gap_df["acc_gap"] = gap_df["train_acc"] - gap_df["test_acc"]
    gap_df["f1_gap"] = gap_df["train_f1"] - gap_df["test_f1"]

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(gap_df)), 5))
    x_positions = list(range(len(gap_df)))
    width = 0.35

    for index, (column, label) in enumerate([("acc_gap", "Acc Gap"), ("f1_gap", "F1 Gap")]):
        offset = (index - 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            gap_df[column],
            width=width,
            label=label,
        )
        annotate_bars(ax, bars, padding=0.003)

    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("BERT Freeze Train-Test Gap")
    ax.set_xlabel("Freeze Strategy")
    ax.set_ylabel("Train - Test")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(gap_df["strategy_label"], rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()

    save_path = Path(output_dir) / "bert_freeze_generalization_gap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] BERT freeze generalization gap figure saved to: {save_path}")


def plot_bert_freeze_efficiency(summary_df, output_dir):
    required = {"trainable_ratio_percent", "test_f1", "strategy_label"}
    if not required <= set(summary_df.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        summary_df["trainable_ratio_percent"],
        summary_df["test_f1"],
        s=80,
        color="#2ca02c",
    )

    for _, row in summary_df.iterrows():
        ax.annotate(
            row["strategy_label"],
            (row["trainable_ratio_percent"], row["test_f1"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.set_title("BERT Freeze Efficiency")
    ax.set_xlabel("Trainable Ratio (%)")
    ax.set_ylabel("Test Macro F1")
    ax.set_ylim(
        max(0, summary_df["test_f1"].min() - 0.04),
        min(1.05, summary_df["test_f1"].max() + 0.04),
    )
    fig.tight_layout()

    save_path = Path(output_dir) / "bert_freeze_efficiency.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] BERT freeze efficiency figure saved to: {save_path}")


def build_bert_fgm_summary_from_configs(configs_dir, layers=FGM_COMPARISON_LAYERS):
    configs_dir = Path(configs_dir)
    rows = []

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    for config_path in sorted(path for path in configs_dir.iterdir() if path.is_dir()):
        config = load_json(config_path / "config.json")
        metrics = load_json(config_path / "metrics.json")
        if not config or not metrics:
            continue

        strategy = config.get("finetune_strategy")
        unfreeze_layers = config.get("unfreeze_last_n_layers")
        if strategy != "partial" or unfreeze_layers is None:
            continue

        unfreeze_layers = int(unfreeze_layers)
        if unfreeze_layers not in layers:
            continue

        run_name = config.get("model_name", config_path.name)
        use_fgm = bool(config.get("use_fgm", False))
        if not use_fgm and "embedding_fgm" in run_name:
            use_fgm = True

        embedding_unfrozen = bool(config.get("embedding_unfrozen", False))
        if use_fgm and "embedding_fgm" in run_name:
            embedding_unfrozen = True

        rows.append(
            {
                "experiment_name": run_name,
                "run_name": config_path.name,
                "finetune_strategy": strategy,
                "unfreeze_last_n_layers": unfreeze_layers,
                "use_fgm": use_fgm,
                "embedding_unfrozen": embedding_unfrozen,
                "train_acc": _metric_value(metrics, "train_acc", "train_accuracy"),
                "train_f1": _metric_value(metrics, "train_f1", "train_macro_f1"),
                "test_acc": _metric_value(metrics, "test_acc", "test_accuracy"),
                "test_f1": _metric_value(metrics, "test_f1", "test_macro_f1"),
                "best_val_f1": _metric_value(metrics, "best_val_f1"),
                "test_loss": _metric_value(metrics, "test_loss"),
                "metrics_saved_at": metrics.get("saved_at"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return prepare_bert_freeze_summary(pd.DataFrame(rows))


def _choose_fgm_variant(group, use_fgm):
    variants = group[group["use_fgm_bool"] == use_fgm].copy()
    if variants.empty:
        return None

    if use_fgm:
        variants["variant_rank"] = variants["embedding_unfrozen_bool"].map({True: 0, False: 1})
    else:
        variants["variant_rank"] = variants["run_name"].astype(str).str.endswith("_no_fgm").map(
            {True: 0, False: 1}
        )

    return variants.sort_values(["variant_rank", "run_name"]).iloc[0]


def build_bert_fgm_comparison(summary_df, output_dir, layers=FGM_COMPARISON_LAYERS):
    output_dir = ensure_dir(output_dir)
    required = {
        "finetune_strategy",
        "unfreeze_last_n_layers",
        "use_fgm",
        "train_acc",
        "train_f1",
        "test_acc",
        "test_f1",
    }
    if not required <= set(summary_df.columns):
        return None

    summary_df = summary_df.copy()
    source_path = Path(output_dir) / "bert_fgm_source_rows.csv"
    missing_path = Path(output_dir) / "bert_fgm_missing_runs.csv"

    fgm_df = summary_df[
        (summary_df["finetune_strategy"] == "partial")
        & (summary_df["unfreeze_last_n_layers"].isin(layers))
    ].copy()
    if fgm_df.empty:
        return None

    fgm_df["use_fgm_bool"] = fgm_df.apply(freeze_row_uses_fgm, axis=1)
    fgm_df["embedding_unfrozen_bool"] = fgm_df.apply(freeze_row_embedding_unfrozen, axis=1)
    if "run_name" not in fgm_df.columns:
        fgm_df["run_name"] = fgm_df.get("experiment_name", fgm_df.index.astype(str))

    fgm_df.sort_values(["unfreeze_last_n_layers", "use_fgm_bool", "run_name"]).to_csv(
        source_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[INFO] BERT FGM source rows saved to: {source_path}")

    comparison_rows = []
    missing_rows = []

    for layer in layers:
        group = fgm_df[fgm_df["unfreeze_last_n_layers"] == layer]
        no_fgm_row = _choose_fgm_variant(group, use_fgm=False)
        fgm_row = _choose_fgm_variant(group, use_fgm=True)

        if no_fgm_row is None:
            missing_rows.append(
                {
                    "unfreeze_last_n_layers": int(layer),
                    "missing_variant": "without_fgm",
                    "expected_run_name": f"bert_partial_last_{layer}_no_fgm",
                }
            )
        if fgm_row is None:
            missing_rows.append(
                {
                    "unfreeze_last_n_layers": int(layer),
                    "missing_variant": "with_fgm",
                    "expected_run_name": f"bert_partial_last_{layer}_embedding_fgm",
                }
            )

        def metric(row, key):
            return None if row is None else _metric_value(row, key)

        def delta(metric_key):
            no_fgm_value = metric(no_fgm_row, metric_key)
            fgm_value = metric(fgm_row, metric_key)
            if no_fgm_value is None or fgm_value is None:
                return None
            return fgm_value - no_fgm_value

        comparison_rows.append(
            {
                "unfreeze_last_n_layers": int(layer),
                "without_fgm_experiment": None if no_fgm_row is None else no_fgm_row["run_name"],
                "with_fgm_experiment": None if fgm_row is None else fgm_row["run_name"],
                "has_without_fgm": no_fgm_row is not None,
                "has_with_fgm": fgm_row is not None,
                "without_fgm_train_acc": metric(no_fgm_row, "train_acc"),
                "with_fgm_train_acc": metric(fgm_row, "train_acc"),
                "delta_train_acc": delta("train_acc"),
                "without_fgm_train_f1": metric(no_fgm_row, "train_f1"),
                "with_fgm_train_f1": metric(fgm_row, "train_f1"),
                "delta_train_f1": delta("train_f1"),
                "without_fgm_test_acc": metric(no_fgm_row, "test_acc"),
                "with_fgm_test_acc": metric(fgm_row, "test_acc"),
                "delta_test_acc": delta("test_acc"),
                "without_fgm_test_f1": metric(no_fgm_row, "test_f1"),
                "with_fgm_test_f1": metric(fgm_row, "test_f1"),
                "delta_test_f1": delta("test_f1"),
            }
        )

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(missing_path, index=False, encoding="utf-8-sig")
        print(f"[WARNING] Missing BERT FGM comparison configs saved to: {missing_path}")
    elif missing_path.exists():
        missing_path.unlink()

    if not comparison_rows:
        return None

    comparison_df = pd.DataFrame(comparison_rows).sort_values("unfreeze_last_n_layers")
    save_path = Path(output_dir) / "bert_fgm_comparison.csv"
    comparison_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] BERT FGM comparison saved to: {save_path}")
    return comparison_df


def plot_bert_fgm_comparison(comparison_df, output_dir):
    if comparison_df is None or comparison_df.empty:
        return

    layer_labels = [f"last {layer}" for layer in comparison_df["unfreeze_last_n_layers"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_positions = list(range(len(comparison_df)))
    width = 0.35

    for ax, metric, title, ylabel in [
        (axes[0], "test_acc", "Embedding FGM Test Accuracy Comparison", "Test Accuracy"),
        (axes[1], "test_f1", "Embedding FGM Test Macro F1 Comparison", "Test Macro F1"),
    ]:
        no_fgm_values = comparison_df[f"without_fgm_{metric}"]
        fgm_values = comparison_df[f"with_fgm_{metric}"]
        no_fgm_bars = ax.bar(
            [x - width / 2 for x in x_positions],
            no_fgm_values,
            width=width,
            label="without_fgm",
        )
        fgm_bars = ax.bar(
            [x + width / 2 for x in x_positions],
            fgm_values,
            width=width,
            label="with_fgm",
        )
        annotate_bars(ax, no_fgm_bars, padding=0.004)
        annotate_bars(ax, fgm_bars, padding=0.004)

        for index, value in enumerate(no_fgm_values):
            if pd.isna(value):
                ax.text(index - width / 2, 0.02, "missing", ha="center", rotation=90, color="#b00020")
        for index, value in enumerate(fgm_values):
            if pd.isna(value):
                ax.text(index + width / 2, 0.02, "missing", ha="center", rotation=90, color="#b00020")

        ax.set_title(title)
        ax.set_xlabel("Partial Unfreeze Layers")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(layer_labels)
        max_value = pd.concat([no_fgm_values, fgm_values]).max(skipna=True)
        ax.set_ylim(0, min(1.08, max_value + 0.08) if pd.notna(max_value) else 1.0)
        ax.legend()

    fig.tight_layout()
    save_path = Path(output_dir) / "bert_fgm_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] BERT FGM comparison figure saved to: {save_path}")

    delta_columns = ["delta_test_acc", "delta_test_f1"]
    if not comparison_df[delta_columns].notna().any().any():
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = list(range(len(comparison_df)))
    for index, (column, label) in enumerate([
        ("delta_test_acc", "Delta Test Acc"),
        ("delta_test_f1", "Delta Test F1"),
    ]):
        offset = (index - 0.5) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            comparison_df[column],
            width=width,
            label=label,
        )
        annotate_bars(ax, bars, padding=0.0008)

    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("FGM Gain Over without_fgm")
    ax.set_xlabel("Partial Unfreeze Layers")
    ax.set_ylabel("with_fgm - without_fgm")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_labels)
    ax.legend()
    fig.tight_layout()

    save_path = Path(output_dir) / "bert_fgm_gain.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] BERT FGM gain figure saved to: {save_path}")


def visualize_bert_freeze_summary(
    summary_csv,
    output_dir,
    comparison_fgm_dir=FGM_COMPARISON_DIR,
    include_fgm_comparison=False,
):
    summary_csv = Path(summary_csv)
    output_dir = ensure_dir(output_dir)
    comparison_fgm_dir = ensure_dir(comparison_fgm_dir)

    if not summary_csv.exists():
        summary_df = build_bert_freeze_summary_from_configs(CONFIGS_DIR)
        if summary_df.empty:
            raise FileNotFoundError(f"BERT freeze summary file not found: {summary_csv}")
        generated_summary_path = output_dir / "bert_freeze_summary.csv"
        summary_df.to_csv(generated_summary_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] BERT freeze summary rebuilt from configs: {generated_summary_path}")
    else:
        summary_df = pd.read_csv(summary_csv)

    if summary_df.empty:
        raise ValueError(f"BERT freeze summary file is empty: {summary_csv}")

    summary_df = prepare_bert_freeze_summary(summary_df)

    configure_plot_style()
    plot_bert_freeze_scores(summary_df, output_dir)
    plot_bert_freeze_trainable(summary_df, output_dir)
    plot_bert_freeze_generalization_gap(summary_df, output_dir)
    plot_bert_freeze_efficiency(summary_df, output_dir)
    if include_fgm_comparison:
        fgm_comparison_df = build_bert_fgm_comparison(summary_df, comparison_fgm_dir)
        plot_bert_fgm_comparison(fgm_comparison_df, comparison_fgm_dir)

    print("\nBERT freeze summary:")
    print(summary_df.to_string(index=False))
    return summary_df


def visualize_bert_fgm_from_configs(configs_dir=CONFIGS_DIR, output_dir=FGM_COMPARISON_DIR):
    output_dir = ensure_dir(output_dir)
    summary_df = build_bert_fgm_summary_from_configs(
        configs_dir=configs_dir,
        layers=FGM_COMPARISON_LAYERS,
    )

    if summary_df.empty:
        raise FileNotFoundError(
            f"No partial-4/8 BERT metrics were found under: {configs_dir}"
        )

    configure_plot_style()
    comparison_df = build_bert_fgm_comparison(summary_df, output_dir)
    plot_bert_fgm_comparison(comparison_df, output_dir)

    print("\nBERT FGM comparison:")
    print(comparison_df.to_string(index=False))
    return comparison_df


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
        "--task",
        choices=["models", "bert_freeze", "fgm", "both"],
        default="models",
        help="Choose whether to visualize model comparison, BERT freeze summary, FGM comparison, or both.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS.keys()),
        help="Model directory names under configs/.",
    )
    parser.add_argument(
        "--parameters-dir",
        default=str(CONFIGS_DIR),
        help="Directory that stores each model's metrics.json/history.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(COMPARISON_DIR),
        help="Directory to save summary files and figures.",
    )
    parser.add_argument(
        "--bert-freeze-summary",
        default=str(OUTPUTS_DIR / "bert_freeze" / "bert_freeze_summary.csv"),
        help="CSV generated by BERT --freeze-sweep.",
    )
    parser.add_argument(
        "--comparison-fgm-dir",
        default=str(FGM_COMPARISON_DIR),
        help="Directory to save partial-4/8 with-vs-without FGM comparison data and figures.",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed_args = parse_args(args)
    model_summary = None
    freeze_summary = None
    fgm_summary = None
    default_comparison_dir = str(COMPARISON_DIR)
    bert_freeze_output_dir = (
        str(OUTPUTS_DIR / "bert_freeze")
        if parsed_args.output_dir == default_comparison_dir
        else parsed_args.output_dir
    )

    if parsed_args.task in {"models", "both"}:
        model_summary = visualize(
            models=parsed_args.models,
            parameters_dir=parsed_args.parameters_dir,
            output_dir=COMPARISON_DIR if parsed_args.output_dir == default_comparison_dir else parsed_args.output_dir,
        )

    if parsed_args.task in {"bert_freeze", "both"}:
        freeze_summary = visualize_bert_freeze_summary(
            summary_csv=parsed_args.bert_freeze_summary,
            output_dir=bert_freeze_output_dir,
            comparison_fgm_dir=parsed_args.comparison_fgm_dir,
        )

    if parsed_args.task in {"fgm", "both"}:
        fgm_summary = visualize_bert_fgm_from_configs(
            configs_dir=parsed_args.parameters_dir,
            output_dir=parsed_args.comparison_fgm_dir,
        )

    if parsed_args.task == "bert_freeze":
        return freeze_summary
    if parsed_args.task == "fgm":
        return fgm_summary
    return model_summary


if __name__ == "__main__":
    main()
