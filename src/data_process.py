import pandas as pd

directory = [
    "data/raw/cnews.test.txt",
    "data/raw/cnews.train.txt",
    "data/raw/cnews.val.txt"
]

target_directory = [
    "data/processed/test_data.csv",
    "data/processed/train_data.csv",
    "data/processed/val_data.csv"
]

for source, target in zip(directory,target_directory):

    txt_path = source

    texts = []
    labels = []

    with open(txt_path, "r", encoding="utf-8") as f:
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