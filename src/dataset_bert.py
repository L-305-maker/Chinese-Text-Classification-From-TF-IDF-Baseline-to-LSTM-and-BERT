import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

try:
    from src.data_processor import build_id_map
    from src.utils.paths import PROCESSED_DATA_DIR
except ModuleNotFoundError:
    from data_processor import build_id_map
    from utils.paths import PROCESSED_DATA_DIR


def load_bert_data():

    train_path = PROCESSED_DATA_DIR / "train_data.csv"
    val_path = PROCESSED_DATA_DIR / "val_data.csv"
    test_path = PROCESSED_DATA_DIR / "test_data.csv"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"找不到文件: {path}")

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    required_cols = {"text", "label"}
    for name, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name}_data 缺少列: {missing}")

    return train_data, val_data, test_data


class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)            # [max_len]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }


def process_loader_bert(model_name="bert-base-chinese", max_len=256, batch_size=16):
    """
    构造 BERT 的 DataLoader
    """
    train_data, val_data, test_data = load_bert_data()

    train_texts = train_data["text"].tolist()
    val_texts = val_data["text"].tolist()
    test_texts = test_data["text"].tolist()

    train_labels = build_id_map(train_data["label"])
    val_labels = build_id_map(val_data["label"])
    test_labels = build_id_map(test_data["label"])

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = BertTextDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_dataset = BertTextDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    test_dataset = BertTextDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = process_loader_bert()

    batch = next(iter(train_loader))

    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("label shape:", batch["label"].shape)
    print("first input_ids:", batch["input_ids"][0][:20])
    print("first attention_mask:", batch["attention_mask"][0][:20])
    print("first label:", batch["label"][0])
