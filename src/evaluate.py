import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.utils.paths import ensure_dir
except ModuleNotFoundError:
    from utils.paths import ensure_dir

from sklearn.metrics import classification_report,confusion_matrix


def save_classification_report(y_true,y_pred,model_name: str,labels=None,target_names=None,save_dir: str = "outputs/metrics"):
    
    #将结果的classification_report保存起来

    ensure_dir(save_dir)

    report = classification_report(y_true,y_pred,labels=labels,target_names=target_names,zero_division=0)

    report_path = os.path.join(save_dir,f"{model_name}_classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[INFO] Classification report saved to: {report_path}")
    print("\n" + report)


def plot_confusion_matrix(y_true,y_pred,model_name: str,labels=None,target_names=None,save_dir: str = "outputs/figures"):

    #画出对应的混淆矩阵

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


def save_predictions(texts,y_true,y_pred,model_name: str,probabilities=None,save_dir: str = "outputs/predictions"):

    #保存对test数据集的预测结果

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


def save_error_analysis(texts,y_true,y_pred,model_name: str,probabilities=None,save_dir: str = "outputs/errors"):
    
    #将错误的预测结果进行保存，方便对模型进行进一步的分析与优化

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
