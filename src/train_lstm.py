import argparse
import random

import torch
import torch.nn as nn

try:
    from models.lstm_classifier import LSTMClassifier
    from src.dataset_lstm import process_loader
    from src.model_utils import (
        initialize_experiment,
        model_dir,
        parameter_dir,
        save_history,
        save_json,
        save_metrics,
        save_torch_checkpoint,
    )
    from src.training_utils import finalize_epoch_stats, init_epoch_stats, update_epoch_stats
except ModuleNotFoundError:
    from dataset_lstm import process_loader
    from model_utils import (
        initialize_experiment,
        model_dir,
        parameter_dir,
        save_history,
        save_json,
        save_metrics,
        save_torch_checkpoint,
    )
    from models.lstm_classifier import LSTMClassifier
    from training_utils import finalize_epoch_stats, init_epoch_stats, update_epoch_stats


MODEL_NAME = "lstm"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, optimizer, criterion, loader, device, grad_clip=1.0):
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

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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
    no_improve_epochs = 0
    patience = config.get("patience", 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    perf = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(epochs):
        train_acc, train_loss, train_f1 = train_one_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            grad_clip=config.get("grad_clip", 1.0),
        )
        val_acc, val_loss, val_f1 = eval_one_epoch(
            model,
            criterion,
            val_loader,
            device,
        )
        scheduler.step(val_f1)

        perf["train_acc"].append(train_acc)
        perf["val_acc"].append(val_acc)
        perf["train_loss"].append(train_loss)
        perf["val_loss"].append(val_loss)
        perf["train_f1"].append(train_f1)
        perf["val_f1"].append(val_f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6g}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   F1: {val_f1:.4f}")
        print("-" * 60)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
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
            print(f"Saved best LSTM model to: {best_path}")
        else:
            no_improve_epochs += 1
            if patience and no_improve_epochs >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    save_history(MODEL_NAME, perf)
    return perf


def build_class_weights(labels, num_classes, device, power=0.5):
    label_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes).float().clamp_min(1.0)
    weights = (counts.sum() / (num_classes * counts)).pow(power)
    return weights.to(device)


def parse_lstm_args(args=None):
    parser = argparse.ArgumentParser(description="LSTM model train")
    parser.add_argument("--embed_dim", "--embed-dim", type=int, default=256)
    parser.add_argument("--hidden_size", "--hidden-size", type=int, default=256)
    parser.add_argument("--num_layers", "--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_len", "--max-len", type=int, default=256)
    parser.add_argument("--min_freq", "--min-freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", "--max-vocab-size", type=int, default=80000)
    parser.add_argument("--tokenizer", choices=["word", "char"], default="word")
    parser.add_argument("--pooling", choices=["last", "mean", "max", "last_mean"], default="last_mean")
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=5e-5)
    parser.add_argument("--grad_clip", "--grad-clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--sample_size", "--sample-size", type=int, default=None)
    parser.add_argument("--class_weight_power", "--class-weight-power", type=float, default=0.5)
    parser.add_argument("--no_class_weights", "--no-class-weights", action="store_true")
    bidirectional_group = parser.add_mutually_exclusive_group()
    bidirectional_group.add_argument("--bidirectional", action="store_true", dest="bidirectional", default=True)
    bidirectional_group.add_argument("--no-bidirectional", action="store_false", dest="bidirectional")
    return parser.parse_args(args)


def main(args=None):
    args = parse_lstm_args(args)
    set_seed(args.seed)

    config = {
        "model_name": MODEL_NAME,
        "num_classes": 10,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "pooling": args.pooling,
        "tokenizer": args.tokenizer,
        "max_len": args.max_len,
        "min_freq": args.min_freq,
        "max_vocab_size": args.max_vocab_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "use_class_weights": not args.no_class_weights,
        "class_weight_power": args.class_weight_power,
        "patience": args.patience,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "criterion": "CrossEntropyLoss",
        "save_model": "checkpoints/lstm/best_model.pth",
    }

    train_loader, val_loader, test_loader, vocab = process_loader(
        batch_size=config["batch_size"],
        tokenizer=config["tokenizer"],
        max_len=config["max_len"],
        min_freq=config["min_freq"],
        max_vocab_size=config["max_vocab_size"],
        num_workers=args.num_workers,
        sample_size=config["sample_size"],
        seed=config["seed"],
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
        pooling=config["pooling"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    class_weights = None
    if config["use_class_weights"]:
        class_weights = build_class_weights(
            labels=train_loader.dataset.label,
            num_classes=config["num_classes"],
            device=device,
            power=config["class_weight_power"],
        )
        config["class_weights"] = class_weights.detach().cpu().tolist()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["label_smoothing"],
    )

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
