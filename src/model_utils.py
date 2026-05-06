import json
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional for JSON conversion.
    np = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PARAMETERS_DIR = PROJECT_ROOT / "parameters"

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


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(model_name):
    return ensure_dir(MODELS_DIR / model_name)


def parameter_dir(model_name):
    return ensure_dir(PARAMETERS_DIR / model_name)


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
