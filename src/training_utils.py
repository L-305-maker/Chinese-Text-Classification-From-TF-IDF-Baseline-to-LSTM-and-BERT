from sklearn.metrics import f1_score

#定义基础数据集，辅助模型
def init_epoch_stats():
    return {
        "loss": 0.0,
        "correct": 0,
        "samples": 0,
        "labels": [],
        "preds": [],
    }


def update_epoch_stats(stats, loss, labels, preds):
    batch_size = labels.size(0)
    stats["loss"] += loss.item() * batch_size
    stats["correct"] += (preds == labels).sum().item()
    stats["samples"] += batch_size
    stats["labels"].extend(labels.detach().cpu().tolist())
    stats["preds"].extend(preds.detach().cpu().tolist())


def finalize_epoch_stats(stats, average="macro"):
    if stats["samples"] == 0:
        raise ValueError("Cannot calculate metrics for an empty epoch.")

    return (
        stats["loss"] / stats["samples"],
        stats["correct"] / stats["samples"],
        f1_score(stats["labels"], stats["preds"], average=average),
    )
