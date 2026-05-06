import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence

from datasets_lstm import process_loader
from model_utils import (
    model_dir,
    parameter_dir,
    save_config,
    save_history,
    save_json,
    save_label_map,
    save_metrics,
    save_torch_checkpoint,
)


MODEL_NAME = "lstm"


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes=10,
        num_layers=1,
        dropout=0.3,
        pad_idx=0,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, input_ids, length):
        x = self.embedding(input_ids)

        packed = pack_padded_sequence(
            x,
            length.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)
        logits = self.linear(final_hidden)
        return logits


def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()

    train_loss = 0.0
    train_samples = 0
    train_correct = 0

    all_label = []
    all_pred = []

    for input_ids, label, _, length in loader:
        input_ids = input_ids.to(device)
        label = label.to(device)
        length = length.to(device)

        optimizer.zero_grad()

        logits = model(input_ids, length)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        pred_label = torch.argmax(logits, dim=1)

        train_loss += loss.item() * label.size(0)
        train_samples += label.size(0)
        train_correct += (pred_label == label).sum().item()

        all_label.extend(label.cpu().tolist())
        all_pred.extend(pred_label.cpu().tolist())

    avg_acc = train_correct / train_samples
    avg_loss = train_loss / train_samples
    avg_f1 = f1_score(all_label, all_pred, average="macro")

    return avg_acc, avg_loss, avg_f1


def eval_one_epoch(model, criterion, loader, device):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    all_pred = []
    all_label = []

    with torch.no_grad():
        for input_ids, label, _, length in loader:
            input_ids = input_ids.to(device)
            label = label.to(device)
            length = length.to(device)

            logits = model(input_ids, length)

            loss = criterion(logits, label)
            preds = torch.argmax(logits, dim=1)

            val_loss += loss.item() * label.size(0)
            val_correct += (preds == label).sum().item()
            val_samples += label.size(0)

            all_pred.extend(preds.cpu().tolist())
            all_label.extend(label.cpu().tolist())

    avg_loss = val_loss / val_samples
    avg_acc = val_correct / val_samples
    avg_f1 = f1_score(all_label, all_pred, average="macro")

    return avg_acc, avg_loss, avg_f1


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs,
    device,
    config,
):
    model_dir(MODEL_NAME)
    parameter_dir(MODEL_NAME)
    save_config(MODEL_NAME, config)
    save_label_map(MODEL_NAME)

    best_val_f1 = float("-inf")

    perf = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": []
    }

    for epoch in range(epochs):
        train_acc, train_loss, train_f1 = train_one_epoch(
            model, optimizer, criterion, train_loader, device
        )
        val_acc, val_loss, val_f1 = eval_one_epoch(
            model, criterion, val_loader, device
        )

        perf["train_acc"].append(train_acc)
        perf["val_acc"].append(val_acc)
        perf["train_loss"].append(train_loss)
        perf["val_loss"].append(val_loss)
        perf["train_f1"].append(train_f1)
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


def main():
    config = {
        "model_name": MODEL_NAME,
        "num_classes": 10,
        "embed_dim": 64,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.3,
        "bidirectional": False,
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 10,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "save_model": "models/lstm/best_model.pth",
    }

    train_loader, val_loader, test_loader, vocab = process_loader(
        batch_size=config["batch_size"]
    )
    config["vocab_size"] = len(vocab)

    save_json(vocab, parameter_dir(MODEL_NAME) / "vocab.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    perf = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        device=device,
        config=config,
    )

    best_model_path = model_dir(MODEL_NAME) / "best_model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_loss, test_f1 = eval_one_epoch(
        model=model,
        criterion=criterion,
        loader=test_loader,
        device=device,
    )

    metrics = {
        "best_val_f1": max(perf["val_f1"]),
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "test_macro_f1": test_f1,
    }
    save_metrics(MODEL_NAME, metrics)

    print("Test Result")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    return perf


if __name__ == "__main__":
    main()
