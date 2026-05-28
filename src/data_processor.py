from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.model_utils import LABEL2ID
    from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
except ModuleNotFoundError:
    from model_utils import LABEL2ID
    from utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


RAW_ONLINE_SHOPPING_FILE = (
    RAW_DATA_DIR / "online_shopping_10_cats" / "online_shopping_10_cats.csv"
)
DATASET_NAME = "online_shopping_10_cats"
DATASET_LABEL_COLUMN = "cat"
DATASET_TEXT_COLUMN = "review"
DATASET_SENTIMENT_COLUMN = "label"
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def read_online_shopping_raw(source=RAW_ONLINE_SHOPPING_FILE):
    source = Path(source)
    source = source if source.exists() else RAW_DATA_DIR / "online_shopping_10_cats.csv"
    if not source.exists():
        raise FileNotFoundError(f"Raw online_shopping_10_cats file not found: {source}")

    raw_data = pd.read_csv(source, encoding="utf-8-sig")
    required_cols = {DATASET_LABEL_COLUMN, DATASET_TEXT_COLUMN}
    missing_cols = required_cols - set(raw_data.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing columns: {missing_cols}")

    data = raw_data[[DATASET_TEXT_COLUMN, DATASET_LABEL_COLUMN]].rename(
        columns={DATASET_LABEL_COLUMN: "label", DATASET_TEXT_COLUMN: "text"}
    )
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    data["label"] = data["label"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    data = data[(data["text"] != "") & (data["label"] != "")].reset_index(drop=True)

    unknown_labels = sorted(set(data["label"]) - set(LABEL2ID))
    if unknown_labels:
        raise ValueError(f"Raw data has unknown labels: {unknown_labels}")

    return data


def process_raw_to_csv(
    train_size=DEFAULT_SPLIT_RATIOS["train"],
    val_size=DEFAULT_SPLIT_RATIOS["val"],
    test_size=DEFAULT_SPLIT_RATIOS["test"],
    random_state=42,
):
    total_size = train_size + val_size + test_size
    if abs(total_size - 1.0) > 1e-8:
        raise ValueError("train_size, val_size, and test_size must add up to 1.0")

    data = read_online_shopping_raw()
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )
    adjusted_val_size = val_size / (train_size + val_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_data["label"],
    )

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, split_data in splits.items():
        target = PROCESSED_DATA_DIR / f"{split}_data.csv"
        split_data = split_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_data.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(split_data)} rows -> {target}")
        print(split_data["label"].value_counts().sort_index())


def load_processed_data(processed_dir=PROCESSED_DATA_DIR):
    data_splits = []
    for split in ["train", "val", "test"]:
        path = processed_dir / f"{split}_data.csv"
        data = pd.read_csv(path, encoding="utf-8-sig")
        missing_cols = {"text", "label"} - set(data.columns)
        if missing_cols:
            raise ValueError(f"{path} is missing columns: {missing_cols}")

        unknown_labels = sorted(set(data["label"]) - set(LABEL2ID))
        if unknown_labels:
            raise ValueError(f"{path} has unknown labels: {unknown_labels}")

        data_splits.append(data)

    return tuple(data_splits)


def data_processor():
    return load_processed_data()


def build_id_map(labels):
    unknown_labels = [label for label in dict.fromkeys(labels) if label not in LABEL2ID]
    if unknown_labels:
        expected = ", ".join(LABEL2ID.keys())
        raise KeyError(f"Unknown labels {unknown_labels!r}. Expected one of: {expected}")
    return [LABEL2ID[label] for label in labels]


if __name__ == "__main__":
    process_raw_to_csv()
