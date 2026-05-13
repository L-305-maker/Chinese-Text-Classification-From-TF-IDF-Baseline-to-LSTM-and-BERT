import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        num_classes: int = 10,
        dropout: float = 0.3,
        finetune_strategy: str = "full",
        unfreeze_last_n_layers: int | None = None
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        self.apply_finetune_strategy(
            finetune_strategy=finetune_strategy,
            unfreeze_last_n_layers=unfreeze_last_n_layers
        )

    def apply_finetune_strategy(
        self,
        finetune_strategy: str,
        unfreeze_last_n_layers: int | None
    ):
        if finetune_strategy == "frozen":
            self.freeze_bert()

        elif finetune_strategy == "partial":
            if unfreeze_last_n_layers is None:
                unfreeze_last_n_layers = 2

            self.freeze_bert()
            self.unfreeze_last_layers(unfreeze_last_n_layers)

        elif finetune_strategy == "full":
            self.unfreeze_bert()

        else:
            raise ValueError(
                f"Unknown finetune_strategy: {finetune_strategy}. "
                f"Expected one of ['frozen', 'partial', 'full']."
            )

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_last_layers(self, n: int):
        total_layers = len(self.bert.encoder.layer)

        if n <= 0:
            raise ValueError("unfreeze_last_n_layers must be greater than 0.")

        if n > total_layers:
            raise ValueError(
                f"unfreeze_last_n_layers={n} is larger than "
                f"the total number of BERT layers={total_layers}."
            )

        for layer in self.bert.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits