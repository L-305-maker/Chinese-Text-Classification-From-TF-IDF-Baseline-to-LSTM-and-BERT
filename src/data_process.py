import pandas as pd

try:
    from src.model_utils import LABEL2ID
    from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
except ModuleNotFoundError:
    from model_utils import LABEL2ID
    from utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def process_raw_to_csv():
    source_files = [
        RAW_DATA_DIR / "cnews.test.txt",
        RAW_DATA_DIR / "cnews.train.txt",
        RAW_DATA_DIR / "cnews.val.txt"
    ]

    target_files = [
        PROCESSED_DATA_DIR / "test_data.csv",
        PROCESSED_DATA_DIR / "train_data.csv",
        PROCESSED_DATA_DIR / "val_data.csv"
    ]

    for source, target in zip(source_files, target_files):
        texts = []
        labels = []

        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    continue

                label, text = parts
                labels.append(label)
                texts.append(text)

        df = pd.DataFrame({
            "text": texts,
            "label": labels
        })

        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False, encoding="utf-8-sig")
        print(df.head())
        print(df["label"].value_counts())


def data_processor():
    test_path = PROCESSED_DATA_DIR / "test_data.csv"
    train_path = PROCESSED_DATA_DIR / "train_data.csv"
    val_path = PROCESSED_DATA_DIR / "val_data.csv"

    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    return train_data, val_data, test_data


def build_id_map(labels):
    ids = []
    for label in labels:
        if label not in LABEL2ID:
            expected = ", ".join(LABEL2ID.keys())
            raise KeyError(f"Unknown label {label!r}. Expected one of: {expected}")
        label_id = LABEL2ID[label]
        ids.append(label_id)

    return ids


if __name__ == "__main__":
    process_raw_to_csv()
