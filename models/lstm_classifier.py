import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        pooling="last_mean",
    ):
        super().__init__()
        if pooling not in {"last", "mean", "max", "last_mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.pooling = pooling

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        classifier_input_dim = lstm_output_dim * 2 if pooling == "last_mean" else lstm_output_dim

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, input_ids, length):
        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)

        packed = pack_padded_sequence(
            x,
            length.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        if self.pooling == "last":
            features = final_hidden
        else:
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=input_ids.size(1),
            )
            mask = (
                torch.arange(input_ids.size(1), device=input_ids.device)
                .unsqueeze(0)
                .lt(length.unsqueeze(1))
            )
            masked_output = output * mask.unsqueeze(-1)

            if self.pooling == "mean":
                features = masked_output.sum(dim=1) / length.unsqueeze(1).clamp_min(1)
            elif self.pooling == "max":
                features = output.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
            else:
                mean_hidden = masked_output.sum(dim=1) / length.unsqueeze(1).clamp_min(1)
                features = torch.cat([final_hidden, mean_hidden], dim=1)

        features = self.dropout(features)
        logits = self.linear(features)
        return logits
