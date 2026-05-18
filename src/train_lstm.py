import torch
import torch.nn as nn
import argparse

try:
    from src.dataset_lstm import process_loader
    from models.lstm_classifier import LSTMClassifier
    from src.training_utils import finalize_epoch_stats, init_epoch_stats, update_epoch_stats
    from src.model_utils import (
        initialize_experiment,
        model_dir,
        parameter_dir,
        save_history,
        save_json,
        save_metrics,
        save_torch_checkpoint,
    )
except ModuleNotFoundError:
    from dataset_lstm import process_loader
    from models.lstm_classifier import LSTMClassifier
    from training_utils import finalize_epoch_stats, init_epoch_stats, update_epoch_stats
    from model_utils import (
        initialize_experiment,
        model_dir,
        parameter_dir,
        save_history,
        save_json,
        save_metrics,
        save_torch_checkpoint,
    )


MODEL_NAME = "lstm"


def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()

    stats = init_epoch_stats()

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
        update_epoch_stats(stats, loss, label, pred_label)

    avg_loss, avg_acc, avg_f1 = finalize_epoch_stats(stats)
    return avg_acc, avg_loss, avg_f1


def eval_one_epoch(model, criterion, loader, device):
    model.eval()

    stats = init_epoch_stats()

    with torch.no_grad():
        for input_ids, label, _, length in loader:
            input_ids = input_ids.to(device)
            label = label.to(device)
            length = length.to(device)

            logits = model(input_ids, length)

            loss = criterion(logits, label)
            preds = torch.argmax(logits, dim=1)
            update_epoch_stats(stats, loss, label, preds)

    avg_loss, avg_acc, avg_f1 = finalize_epoch_stats(stats)
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
    initialize_experiment(MODEL_NAME, config)

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

def parse_lstm_args(args=None):
    parser = argparse.ArgumentParser(description="LSTM model train")
    parser.add_argument("--embed_dim",type=int,default=64)
    parser.add_argument("--hidden_size",type=int,default=512)
    parser.add_argument("--num_layers",type=int,default=1)
    parser.add_argument("--dropout",type=float,default=0.3)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--learning_rate",type=float,default=0.001)
    parser.add_argument("--epochs",type=int,default=5)

    return parser.parse_args(args)


def main(args=None):
    args = parse_lstm_args(args)
    config = {
        "model_name": MODEL_NAME,
        "num_classes": 10,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": False,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "save_model": "checkpoints/lstm/best_model.pth",
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
