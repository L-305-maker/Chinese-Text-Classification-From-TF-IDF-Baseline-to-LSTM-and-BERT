import pandas as pd
from pathlib import Path


def process_raw_to_csv():
    current_script_dir = Path(__file__).resolve().parent
    base_path = current_script_dir.parent

    source_files = [
        base_path / "data" / "raw" / "cnews.test.txt",
        base_path / "data" / "raw" / "cnews.train.txt",
        base_path / "data" / "raw" / "cnews.val.txt"
    ]

    target_files = [
        base_path / "data" / "processed" / "test_data.csv",
        base_path / "data" / "processed" / "train_data.csv",
        base_path / "data" / "processed" / "val_data.csv"
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

        df.to_csv(target, index=False, encoding="utf-8-sig")
        print(df.head())
        print(df["label"].value_counts())


def data_processor():
    current_script_dir = Path(__file__).resolve().parent
    base_path = current_script_dir.parent / "data" / "processed"

    test_path = base_path / "test_data.csv"
    train_path = base_path / "train_data.csv"
    val_path = base_path / "val_data.csv"

    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    return train_data,test_data, val_data


def build_id_map(labels):
    label2id = {
        "体育": 0,
        "财经": 1,
        "娱乐": 2,
        "家居": 3,
        "房产": 4,
        "教育": 5,
        "时尚": 6,
        "时政": 7,
        "游戏": 8,
        "科技": 9
    }

    ids = []
    for label in labels:
        label_id = label2id[label]
        ids.append(label_id)

    return ids


if __name__ == "__main__":
    process_raw_to_csv()