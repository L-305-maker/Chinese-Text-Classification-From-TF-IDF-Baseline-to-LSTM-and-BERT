from collections import Counter

import jieba
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from src.data_processor import build_id_map, data_processor
except ModuleNotFoundError:
    from data_processor import build_id_map, data_processor


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1


def tokenize(text, tokenizer="word"):
    text = str(text)
    if tokenizer == "char":
        return list(text)
    return jieba.lcut(text)


def build_vocab(texts, tokenizer="word", min_freq=2, max_vocab_size=None):
    counter = Counter()
    for sentence in texts:
        counter.update(tokenize(sentence, tokenizer=tokenizer))

    vocab = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
    }

    for token, count in counter.most_common(max_vocab_size):
        if count < min_freq:
            continue
        vocab[token] = len(vocab)

    return vocab


def sample_train_data(train_data, sample_size, seed=42):
    if sample_size is None or sample_size >= len(train_data):
        return train_data

    label_count = train_data["label"].nunique()
    per_label = max(1, sample_size // label_count)
    sampled_parts = [
        group.sample(n=min(len(group), per_label), random_state=seed)
        for _, group in train_data.groupby("label", sort=False)
    ]
    sampled_df = pd.concat(sampled_parts, axis=0)

    remaining = sample_size - len(sampled_df)
    if remaining > 0:
        rest = train_data.drop(sampled_df.index)
        extra = rest.sample(n=min(remaining, len(rest)), random_state=seed)
        sampled_df = pd.concat([sampled_df, extra], axis=0)

    return sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def encode(vocab, text, tokenizer="word", max_len=512):
    tokens = tokenize(text, tokenizer=tokenizer)
    if max_len is not None:
        tokens = tokens[:max_len]

    ids = [vocab.get(token, UNK_ID) for token in tokens]
    return ids or [UNK_ID]


def collate_fn(batch):
    text = [item[0] for item in batch]
    label = [item[1] for item in batch]

    max_len = max(len(sentence) for sentence in text)
    ids_list = []
    mask_list = []
    length_list = []

    for sentence in text:
        length = max(1, len(sentence))
        padding_len = max_len - len(sentence)
        padded_ids = sentence + [PAD_ID] * padding_len
        attention_mask = [1] * len(sentence) + [0] * padding_len

        ids_list.append(padded_ids)
        mask_list.append(attention_mask)
        length_list.append(length)

    ids_tensor = torch.tensor(ids_list, dtype=torch.long)
    label_tensor = torch.tensor(label, dtype=torch.long)
    mask_tensor = torch.tensor(mask_list, dtype=torch.float)
    length_tensor = torch.tensor(length_list, dtype=torch.long)
    return ids_tensor, label_tensor, mask_tensor, length_tensor


class LSTMdataset(Dataset):
    def __init__(self, data, vocab, tokenizer="word", max_len=512):
        self.label = build_id_map(data["label"])
        self.input_ids = [
            encode(vocab, text, tokenizer=tokenizer, max_len=max_len)
            for text in data["text"].tolist()
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.label[index]


def process_loader(
    batch_size=16,
    tokenizer="word",
    max_len=512,
    min_freq=2,
    max_vocab_size=None,
    num_workers=0,
    sample_size=None,
    seed=42,
):
    train_data, val_data, test_data = data_processor()
    train_data = sample_train_data(train_data, sample_size=sample_size, seed=seed)

    vocab = build_vocab(
        train_data["text"],
        tokenizer=tokenizer,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )

    train_dataset = LSTMdataset(train_data, vocab, tokenizer=tokenizer, max_len=max_len)
    val_dataset = LSTMdataset(val_data, vocab, tokenizer=tokenizer, max_len=max_len)
    test_dataset = LSTMdataset(test_data, vocab, tokenizer=tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, vocab
