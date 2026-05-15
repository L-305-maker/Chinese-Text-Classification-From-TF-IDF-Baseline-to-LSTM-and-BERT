import json
from datetime import datetime
from pathlib import Path
from torch.optim import AdamW

try:
    from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, PROJECT_ROOT, ensure_dir
except ModuleNotFoundError:
    from utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, PROJECT_ROOT, ensure_dir

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional for JSON conversion.
    np = None


MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = CHECKPOINTS_DIR
CONFIG_DIR = CONFIGS_DIR

LABEL2ID = {
    "体育": 0,
    "财经": 1,
    "娱乐": 2,
    "家居": 3,
    "房产": 4,
    "教育": 5,
    "时尚": 6,
    "时政": 7,
    "游戏": 8,
    "科技": 9,
}

ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def model_dir(model_name):
    return checkpoint_dir(model_name)


def parameter_dir(model_name):
    return config_dir(model_name)


def checkpoint_dir(model_name):
    return ensure_dir(CHECKPOINT_DIR / model_name)


def config_dir(model_name):
    return ensure_dir(CONFIG_DIR / model_name)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=4)

    return path


def save_label_map(model_name):
    path = parameter_dir(model_name) / "label_map.json"
    return save_json(
        {
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "saved_at": now_iso(),
        },
        path,
    )


def save_config(model_name, config):
    payload = {
        **config,
        "saved_at": now_iso(),
    }
    return save_json(payload, parameter_dir(model_name) / "config.json")


def save_metrics(model_name, metrics):
    payload = {
        **metrics,
        "saved_at": now_iso(),
    }
    return save_json(payload, parameter_dir(model_name) / "metrics.json")


def save_history(model_name, history):
    payload = {
        "history": history,
        "saved_at": now_iso(),
    }
    return save_json(payload, parameter_dir(model_name) / "history.json")


def save_sklearn_model(model_name, model, filename="model.pkl"):
    import joblib

    path = model_dir(model_name) / filename
    joblib.dump(model, path)
    return path


def save_torch_checkpoint(
    model_name,
    model,
    optimizer=None,
    epoch=None,
    config=None,
    metrics=None,
    filename="best_model.pth",
):
    import torch

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": config or {},
        "metrics": metrics or {},
        "saved_at": now_iso(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    path = model_dir(model_name) / filename
    torch.save(checkpoint, path)
    return path


def count_parameters(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = trainable_params / total_params if total_params>0 else 0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio
    }

def print_trainable_parameters(model):
    param_info = count_parameters(model)

    print(f"Total parameters:{param_info['total_params']}")
    print(f"Trainable parameters:{param_info['trainable_params']}")
    print(f"Trainable Ratio:{param_info['trainable_ratio']*100:.2f}%")

    return param_info


def build_optimizer(
    model,
    bert_lr: float = 2e-5,
    classifier_lr: float = 1e-4,
    weight_decay: float = 0.01
):

    bert_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("bert."):
            bert_params.append(param)
        else:
            classifier_params.append(param)

    optimizer_grouped_parameters = []

    if len(bert_params) > 0:
        optimizer_grouped_parameters.append({
            "params": bert_params,
            "lr": bert_lr,
            "weight_decay": weight_decay
        })

    if len(classifier_params) > 0:
        optimizer_grouped_parameters.append({
            "params": classifier_params,
            "lr": classifier_lr,
            "weight_decay": weight_decay
        })

    optimizer = AdamW(optimizer_grouped_parameters)

    print(f"BERT trainable parameter tensors:       {len(bert_params)}")
    print(f"Classifier trainable parameter tensors: {len(classifier_params)}")
    print(f"BERT learning rate:                     {bert_lr}")
    print(f"Classifier learning rate:               {classifier_lr}")

    return optimizer
