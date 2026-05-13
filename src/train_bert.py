import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import BertModel
import argparse
from pathlib import Path
import pandas as pd
from models.bert.bert_unfreeze_last_n_layers import BertClassifier
from src.model_utils import print_trainable_parameters,build_optimizer

try:
    from src.adversarial import FGM
    from src.datasets_bert import process_loader_bert
    from src.model_utils import (
        model_dir,
        parameter_dir,
        save_config,
        save_history,
        save_label_map,
        save_metrics,
        save_torch_checkpoint,
        ID2LABEL,
    )
except ModuleNotFoundError:
    from adversarial import FGM
    from datasets_bert import process_loader_bert
    from model_utils import (
        model_dir,
        parameter_dir,
        save_config,
        save_history,
        save_label_map,
        save_metrics,
        save_torch_checkpoint,
        ID2LABEL,
    )


MODEL_NAME = "bert"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARTIAL_UNFREEZE_LAYERS = [1, 2, 4, 8, 12]
FGM_PARTIAL_LAYERS = {4, 8}


'''class BertClassifier(nn.Module):
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

        return logits'''


def train_one_epoch(model, loader, optimizer, criterion, device, fgm=None):
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

        model.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()

        if fgm is not None:
            attacked = fgm.attack()
            if attacked > 0:
                try:
                    adv_logits = model(input_ids, attention_mask)
                    adv_loss = criterion(adv_logits, labels)
                    adv_loss.backward()
                finally:
                    fgm.restore()

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


def predict_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            probabilities = F.softmax(logits, dim=1)
            confidence, preds = torch.max(probabilities, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probabilities.extend(confidence.cpu().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "macro_f1": avg_f1,
        "y_true": all_labels,
        "y_pred": all_preds,
        "probabilities": all_probabilities,
    }


def best_epoch_index(perf):
    val_f1 = perf.get("val_f1", [])
    if not val_f1:
        return None
    return max(range(len(val_f1)), key=val_f1.__getitem__)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    config,
    model_name=MODEL_NAME,
    fgm=None,
):
    model_dir(model_name)
    parameter_dir(model_name)
    save_config(model_name, config)
    save_label_map(model_name)

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
            model, train_loader, optimizer, criterion, device, fgm=fgm
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
                model_name=model_name,
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

    save_history(model_name, perf)
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

def args_bert_parse(args=None):
    parser = argparse.ArgumentParser(description="BERT model train")
    parser.add_argument("--dropout",type=float,default=0.3)
    parser.add_argument("--max_len",type=int,default=256)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument(
        "--unfreeze_last_n_layers",
        type=int,
        default=None,
        help="Number of last BERT layers to unfreeze when using partial fine-tuning"
    )
    parser.add_argument("--bert_lr",type=float,default=2e-5,help="Learning rate for BERT parameters")
    parser.add_argument("--classifier_lr",type=float,default=1e-4,help="Learning rate for classifier head")
    parser.add_argument("--weight_decay",type=float,default=0.01,help="Weight decay for AdamW optimizer")
    parser.add_argument(
        "--use_fgm",
        "--use-fgm",
        action="store_true",
        dest="use_fgm",
        help="Use Partial-4/8 + Embedding Unfrozen + FGM experiments."
    )
    parser.add_argument(
        "--fgm_epsilon",
        "--fgm-epsilon",
        type=float,
        default=1.0,
        dest="fgm_epsilon",
        help="FGM perturbation scale."
    )
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        default="full",
        choices=["full","frozen","partial"],
        help="choose the bert fine-tuning strategy"
    )
    parser.add_argument(
        "--freeze-sweep",
        action="store_true",
        dest="freeze_sweep",
        help="Run BERT experiments across frozen, partial, and full fine-tuning strategies."
    )
    parser.add_argument(
        "--partial_unfreeze_layers",
        "--partial-unfreeze-layers",
        nargs="+",
        type=int,
        default=DEFAULT_PARTIAL_UNFREEZE_LAYERS,
        dest="partial_unfreeze_layers",
        help="Layer counts used by --freeze-sweep for partial fine-tuning."
    )
    parser.add_argument(
        "--freeze-output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        dest="freeze_output_dir",
        help="Directory for BERT freeze summary CSV and evaluation artifacts."
    )
    parser.add_argument(
        "--freeze-summary-filename",
        type=str,
        default="bert_freeze_summary.csv",
        dest="freeze_summary_filename",
        help="CSV filename for the BERT freeze experiment summary."
    )
    parser.add_argument(
        "--skip-freeze-visualize",
        action="store_true",
        dest="skip_freeze_visualize",
        help="Skip visualization after saving the BERT freeze summary CSV."
    )
    parsed_args = parser.parse_args(args)
    if parsed_args.finetune_strategy == "partial" and parsed_args.unfreeze_last_n_layers is None:
        parsed_args.unfreeze_last_n_layers = 2
    return parsed_args


def fgm_is_supported(args):
    return (
        args.finetune_strategy == "partial"
        and args.unfreeze_last_n_layers in FGM_PARTIAL_LAYERS
    )


def embedding_unfrozen_for_fgm(args):
    return args.use_fgm and fgm_is_supported(args)


def unfreeze_bert_embeddings(model):
    matched = False
    for name, param in model.named_parameters():
        if name.startswith("bert.embeddings."):
            matched = True
            param.requires_grad_(True)

    if not matched:
        raise ValueError("No BERT embedding parameters found to unfreeze.")


def validate_fgm_args(args):
    if args.use_fgm and not fgm_is_supported(args):
        allowed = ", ".join(str(layer) for layer in sorted(FGM_PARTIAL_LAYERS))
        raise ValueError(
            "FGM is only supported for partial fine-tuning with "
            f"unfreeze_last_n_layers in {{{allowed}}}."
        )


def fgm_suffix(args):
    return "embedding_fgm" if args.use_fgm else "no_fgm"


def build_fgm(args, model):
    validate_fgm_args(args)
    if not args.use_fgm:
        return None

    fgm = FGM(model=model, epsilon=args.fgm_epsilon)
    fgm.enable_attack_grad()
    print(f"FGM adversarial training enabled, epsilon={args.fgm_epsilon}")
    return fgm


def build_Bert_model(args,num_classes,device):
    model = BertClassifier(
        model_name="bert-base-chinese",
        num_classes = num_classes,
        dropout = args.dropout,
        finetune_strategy=args.finetune_strategy,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers
    ).to(device)

    if embedding_unfrozen_for_fgm(args):
        unfreeze_bert_embeddings(model)
        print("Embedding unfrozen for Partial-4/8 + FGM experiment.")

    param_info = print_trainable_parameters(model)

    return model,param_info

def build_Bert_optimizer(args,model):
    optimizer = build_optimizer(
        model=model,
        bert_lr=args.bert_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay
    )

    return optimizer


def build_bert_config(args, model_name=MODEL_NAME):
    return {
        "model_name": model_name,
        "pretrained_model": "bert-base-chinese",
        "num_classes": 10,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "bert_lr":args.bert_lr,
        "classifier_lr":args.classifier_lr,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "criterion": "CrossEntropyLoss",
        "save_model": f"models/{model_name}/best_model.pth",
        "weight_decay":args.weight_decay,
        "finetune_strategy":args.finetune_strategy,
        "unfreeze_last_n_layers":args.unfreeze_last_n_layers,
        "use_fgm": args.use_fgm,
        "fgm_epsilon": args.fgm_epsilon,
        "embedding_unfrozen": embedding_unfrozen_for_fgm(args),
    }


def build_freeze_run_name(finetune_strategy, unfreeze_last_n_layers=None):
    if finetune_strategy == "partial":
        return f"bert_partial_last_{unfreeze_last_n_layers}"
    return f"bert_{finetune_strategy}"


def build_experiment_name(args):
    freeze_name = build_freeze_run_name(
        finetune_strategy=args.finetune_strategy,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    )
    return f"{freeze_name}_{fgm_suffix(args)}"


def freeze_sweep_specs(partial_unfreeze_layers, include_fgm=False):
    layers = {int(layer) for layer in partial_unfreeze_layers}
    invalid_layers = [layer for layer in layers if layer <= 0]
    if invalid_layers:
        raise ValueError(f"partial_unfreeze_layers must be positive integers: {invalid_layers}")

    if include_fgm:
        layers.update(FGM_PARTIAL_LAYERS)

    specs = [("frozen", None, False)]
    for layer in sorted(layers):
        specs.append(("partial", layer, False))
        if include_fgm and layer in FGM_PARTIAL_LAYERS:
            specs.append(("partial", layer, True))
    specs.append(("full", None, False))
    return specs


def clone_args_for_strategy(args, finetune_strategy, unfreeze_last_n_layers, use_fgm):
    run_args = argparse.Namespace(**vars(args))
    run_args.finetune_strategy = finetune_strategy
    run_args.unfreeze_last_n_layers = unfreeze_last_n_layers
    run_args.use_fgm = use_fgm
    return run_args


def save_bert_evaluation_outputs(model_name, test_result, test_loader, output_dir):
    try:
        from src.evaluate import (
            plot_confusion_matrix,
            save_classification_report,
            save_error_analysis,
            save_predictions,
        )
    except ModuleNotFoundError:
        from evaluate import (
            plot_confusion_matrix,
            save_classification_report,
            save_error_analysis,
            save_predictions,
        )

    output_dir = Path(output_dir)
    labels = sorted(ID2LABEL)
    target_names = [ID2LABEL[label] for label in labels]
    texts = getattr(test_loader.dataset, "texts", None)

    save_classification_report(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        model_name=model_name,
        labels=labels,
        target_names=target_names,
        save_dir=str(output_dir / "bert_freeze_classification_reports"),
    )
    plot_confusion_matrix(
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        model_name=model_name,
        labels=labels,
        target_names=target_names,
        save_dir=str(output_dir / "bert_freeze_confusion_matrices"),
    )

    if texts is not None:
        save_predictions(
            texts=texts,
            y_true=test_result["y_true"],
            y_pred=test_result["y_pred"],
            model_name=model_name,
            probabilities=test_result["probabilities"],
            save_dir=str(output_dir / "bert_freeze_predictions"),
        )
        save_error_analysis(
            texts=texts,
            y_true=test_result["y_true"],
            y_pred=test_result["y_pred"],
            model_name=model_name,
            probabilities=test_result["probabilities"],
            save_dir=str(output_dir / "bert_freeze_wrong_cases"),
        )


def build_freeze_summary_row(
    model_name,
    args,
    param_info,
    perf,
    train_result,
    test_result,
    checkpoint_path,
):
    best_index = best_epoch_index(perf)

    return {
        "experiment_name": model_name,
        "run_name": model_name,
        "finetune_strategy": args.finetune_strategy,
        "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "use_fgm": args.use_fgm,
        "fgm_epsilon": args.fgm_epsilon,
        "embedding_unfrozen": embedding_unfrozen_for_fgm(args),
        "trainable_ratio": param_info["trainable_ratio"],
        "trainable_ratio_percent": param_info["trainable_ratio"] * 100,
        "trainable_params": param_info["trainable_params"],
        "total_params": param_info["total_params"],
        "train_loss": train_result["loss"],
        "train_acc": train_result["accuracy"],
        "train_f1": train_result["macro_f1"],
        "test_loss": test_result["loss"],
        "test_acc": test_result["accuracy"],
        "test_f1": test_result["macro_f1"],
        "best_epoch_by_val_f1": best_index + 1 if best_index is not None else None,
        "best_val_loss": perf["val_loss"][best_index] if best_index is not None else None,
        "best_val_acc": perf["val_acc"][best_index] if best_index is not None else None,
        "best_val_f1": perf["val_f1"][best_index] if best_index is not None else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "bert_lr": args.bert_lr,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "checkpoint_path": str(checkpoint_path),
    }


def run_bert_experiment(
    args,
    train_loader,
    val_loader,
    test_loader,
    tokenizer,
    device,
    model_name=MODEL_NAME,
    save_detail_outputs=False,
    output_dir=None,
):
    config = build_bert_config(args, model_name=model_name)

    tokenizer_dir = model_dir(model_name) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    config["tokenizer_dir"] = str(tokenizer_dir)

    model,param_info = build_Bert_model(args,config["num_classes"],device)
    config.update(param_info)

    optimizer = build_Bert_optimizer(args,model)
    fgm = build_fgm(args, model)
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
        model_name=model_name,
        fgm=fgm,
    )

    best_model_path = model_dir(model_name) / "best_model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_result = predict_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
    )
    test_result = predict_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics = {
        **param_info,
        "best_val_f1": max(perf["val_f1"]),
        "train_loss": train_result["loss"],
        "train_accuracy": train_result["accuracy"],
        "train_macro_f1": train_result["macro_f1"],
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_macro_f1": test_result["macro_f1"],
    }
    save_metrics(model_name, metrics)

    print("Train Result")
    print(
        f"Train Loss: {train_result['loss']:.4f}, "
        f"Train Acc: {train_result['accuracy']:.4f}, "
        f"Train F1: {train_result['macro_f1']:.4f}"
    )
    print("Test Result")
    print(
        f"Test Loss: {test_result['loss']:.4f}, "
        f"Test Acc: {test_result['accuracy']:.4f}, "
        f"Test F1: {test_result['macro_f1']:.4f}"
    )
    print("-" * 60)

    if save_detail_outputs and output_dir is not None:
        save_bert_evaluation_outputs(
            model_name=model_name,
            test_result=test_result,
            test_loader=test_loader,
            output_dir=output_dir,
        )

    summary_row = build_freeze_summary_row(
        model_name=model_name,
        args=args,
        param_info=param_info,
        perf=perf,
        train_result=train_result,
        test_result=test_result,
        checkpoint_path=best_model_path,
    )

    return perf, summary_row


def run_freeze_sweep(args):
    config = build_bert_config(args)
    train_loader, val_loader, test_loader, tokenizer = process_loader_bert(
        model_name=config["pretrained_model"],
        max_len=config["max_len"],
        batch_size=config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.freeze_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for finetune_strategy, unfreeze_last_n_layers, use_fgm in freeze_sweep_specs(
        args.partial_unfreeze_layers,
        include_fgm=args.use_fgm,
    ):
        run_args = clone_args_for_strategy(
            args=args,
            finetune_strategy=finetune_strategy,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            use_fgm=use_fgm,
        )
        run_name = build_experiment_name(run_args)

        print(f"\nRunning BERT freeze experiment: {run_name}")
        print("=" * 60)

        _, summary_row = run_bert_experiment(
            args=run_args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            device=device,
            model_name=run_name,
            save_detail_outputs=True,
            output_dir=output_dir,
        )
        rows.append(summary_row)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / args.freeze_summary_filename
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] BERT freeze summary saved to: {summary_path}")

    if not args.skip_freeze_visualize:
        try:
            from src.visualize import visualize_bert_freeze_summary
        except ModuleNotFoundError:
            from visualize import visualize_bert_freeze_summary

        visualize_bert_freeze_summary(summary_csv=summary_path, output_dir=output_dir)

    print("\nBERT freeze summary:")
    print(summary_df.to_string(index=False))
    return summary_df


def main(args=None):
    args = args_bert_parse(args)

    if args.freeze_sweep:
        return run_freeze_sweep(args)

    validate_fgm_args(args)
    config = build_bert_config(args)

    train_loader, val_loader, test_loader, tokenizer = process_loader_bert(
        model_name=config["pretrained_model"],
        max_len=config["max_len"],
        batch_size=config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perf, _ = run_bert_experiment(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        model_name=build_experiment_name(args),
        save_detail_outputs=False,
    )

    return perf


if __name__ == "__main__":
    main()
