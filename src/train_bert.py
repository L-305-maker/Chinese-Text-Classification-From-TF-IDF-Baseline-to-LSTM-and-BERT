import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import BertModel

from datasets_bert import process_loader_bert
from model_utils import (
    model_dir,
    parameter_dir,
    save_config,
    save_history,
    save_label_map,
    save_metrics,
    save_torch_checkpoint,
)


MODEL_NAME = "bert"


class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_classes=10, dropout=0.3):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)

        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, avg_acc, avg_f1


def eval_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, avg_acc, avg_f1


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, config):
    model_dir(MODEL_NAME)
    parameter_dir(MODEL_NAME)
    save_config(MODEL_NAME, config)
    save_label_map(MODEL_NAME)

    best_val_f1 = float("-inf")

    perf = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, val_f1 = eval_one_epoch(
            model, val_loader, criterion, device
        )

        perf["train_loss"].append(train_loss)
        perf["train_acc"].append(train_acc)
        perf["train_f1"].append(train_f1)
        perf["val_loss"].append(val_loss)
        perf["val_acc"].append(val_acc)
        perf["val_f1"].append(val_f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   F1: {val_f1:.4f}")
        print("-" * 60)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = save_torch_checkpoint(
                model_name=MODEL_NAME,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                config=config,
                metrics={
                    "best_val_f1": best_val_f1,
                    "best_val_acc": val_acc,
                    "best_val_loss": val_loss,
                },
                filename="best_model.pth",
            )
            print(f"保存当前最优模型到: {best_path}")

    save_history(MODEL_NAME, perf)
    return perf


def test_model(model, test_loader, criterion, device):
    test_loss, test_acc, test_f1 = eval_one_epoch(model, test_loader, criterion, device)

    print("Test Result")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print("-" * 60)

    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }


def main():
    config = {
        "model_name": MODEL_NAME,
        "pretrained_model": "bert-base-chinese",
        "num_classes": 10,
        "dropout": 0.3,
        "max_len": 128,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 5,
        "optimizer": "AdamW",
        "criterion": "CrossEntropyLoss",
        "save_model": "models/bert/best_model.pth",
    }

    train_loader, val_loader, test_loader, tokenizer = process_loader_bert(
        model_name=config["pretrained_model"],
        max_len=config["max_len"],
        batch_size=config["batch_size"]
    )

    tokenizer_dir = model_dir(MODEL_NAME) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    config["tokenizer_dir"] = str(tokenizer_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertClassifier(
        model_name=config["pretrained_model"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    perf = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config["epochs"],
        config=config,
    )

    best_model_path = model_dir(MODEL_NAME) / "best_model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    save_metrics(
        MODEL_NAME,
        {
            "best_val_f1": max(perf["val_f1"]),
            **test_metrics,
        },
    )

    return perf


if __name__ == "__main__":
    main()
