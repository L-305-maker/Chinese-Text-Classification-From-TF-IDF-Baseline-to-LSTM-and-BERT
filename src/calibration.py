import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.utils.paths import OUTPUTS_DIR, ensure_dir


MPL_CONFIG_DIR = ensure_dir(OUTPUTS_DIR / ".matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []

    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        logits.append(model(**inputs).cpu())
        labels.append(batch["label"].cpu())

    return torch.cat(logits), torch.cat(labels)


def logits_metrics(logits, labels):
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return {
        "nll": float(F.cross_entropy(logits, labels).item()),
        "accuracy": float((preds == labels).float().mean().item()),
    }


def expected_calibration_error(logits, labels, n_bins=10):
    probs = F.softmax(logits, dim=1)
    confidence, preds = probs.max(dim=1)
    correct = preds.eq(labels)
    edges = torch.linspace(0, 1, n_bins + 1)
    total = labels.numel()
    ece = torch.tensor(0.0)
    rows = []

    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidence >= lower) & (confidence <= upper) if index == 0 else ( (confidence > lower) & (confidence <= upper) )
        count = int(in_bin.sum().item())
        row = {
            "bin": index + 1,
            "lower": float(lower.item()),
            "upper": float(upper.item()),
            "count": count,
            "accuracy": None,
            "confidence": None,
            "gap": None,
        }
        if count:
            bin_acc = correct[in_bin].float().mean()
            bin_conf = confidence[in_bin].mean()
            gap = (bin_acc - bin_conf).abs()
            ece += gap * (count / total)
            row.update(
                {
                    "accuracy": float(bin_acc.item()),
                    "confidence": float(bin_conf.item()),
                    "gap": float(gap.item()),
                }
            )
        rows.append(row)

    return float(ece.item()), rows


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, max_iter=500):
        logits, labels = logits.detach(), labels.detach()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


def plot_reliability(before_bins, after_bins, save_path, title):
    def points(rows):
        rows = [row for row in rows if row["count"]]
        return [row["confidence"] for row in rows], [row["accuracy"] for row in rows]

    ensure_dir(Path(save_path).parent)
    before_conf, before_acc = points(before_bins)
    after_conf, after_acc = points(after_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="perfect")
    ax.plot(before_conf, before_acc, marker="o", label="before")
    ax.plot(after_conf, after_acc, marker="o", label="after")
    ax.set_title(title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Reliability diagram saved to: {save_path}")


def run_temperature_scaling_calibration(model,val_loader,test_loader,device,model_name,n_bins=10,output_dir=OUTPUTS_DIR / "calibration",test_logits=None,test_labels=None,):
    output_dir = ensure_dir(output_dir)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    if test_logits is None or test_labels is None:
        test_logits, test_labels = collect_logits(model, test_loader, device)

    before_ece, before_bins = expected_calibration_error(test_logits, test_labels, n_bins)
    before = logits_metrics(test_logits, test_labels)

    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)
    calibrated_logits = scaler(test_logits).detach()

    after_ece, after_bins = expected_calibration_error(calibrated_logits, test_labels, n_bins)
    after = logits_metrics(calibrated_logits, test_labels)

    results = {
        "model_name": model_name,
        "temperature": temperature,
        "n_bins": int(n_bins),
        "ece_before": before_ece,
        "ece_after": after_ece,
        "nll_before": before["nll"],
        "nll_after": after["nll"],
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"]
    }

    metrics_path = output_dir / f"{model_name}_calibration_metrics.json"
    bins_path = output_dir / f"{model_name}_calibration_bins.csv"
    figure_path = output_dir / f"{model_name}_reliability_diagram.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    pd.DataFrame(
        [{**row, "stage": "before"} for row in before_bins]
        + [{**row, "stage": "after"} for row in after_bins]
    ).to_csv(bins_path, index=False, encoding="utf-8-sig")

    plot_reliability(
        before_bins=before_bins,
        after_bins=after_bins,
        save_path=figure_path,
        title=f"Reliability Diagram - {model_name}",
    )

    print(f"[INFO] Calibration metrics saved to: {metrics_path}")
    print(f"[INFO] Calibration bins saved to: {bins_path}")
    return {
        **results,
        "metrics_path": str(metrics_path),
        "bins_path": str(bins_path),
        "figure_path": str(figure_path),
    }
