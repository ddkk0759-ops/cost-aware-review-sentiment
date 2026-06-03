"""PyTorch sequence-model definitions: MeanPool / LSTM / Transformer / TextCNN."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import config


def _seq_dropout() -> float:
    return float(config.SEQ_DROPOUT)


def _tfm_dropout() -> float:
    return float(config.SEQ_TRANSFORMER_DROPOUT)


class MeanPoolingModel(nn.Module):
    def __init__(self, embedding_matrix, pad_idx: int = 0, num_classes: int = 2):
        super().__init__()
        self.pad_idx = pad_idx
        emb = torch.as_tensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=pad_idx)
        embed_dim = emb.shape[1]
        d = _seq_dropout()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.fc(pooled)


class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, pad_idx: int = 0,
                 hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        emb = torch.as_tensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=pad_idx)
        embed_dim = emb.shape[1]
        d = _seq_dropout()
        lstm_drop = d if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=lstm_drop, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        emb = self.embedding(x)
        _, (hn, _) = self.lstm(emb)
        out = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(out)


class _PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float | None = None):
        super().__init__()
        if dropout is None:
            dropout = _tfm_dropout()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self, embedding_matrix, pad_idx: int = 0,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, num_classes: int = 2,
                 dropout: float | None = None):
        super().__init__()
        self.pad_idx = pad_idx
        if dropout is None:
            dropout = _tfm_dropout()
        emb = torch.as_tensor(embedding_matrix)
        embed_dim = emb.shape[1]
        assert embed_dim % nhead == 0, "embed_dim 必须能被 nhead 整除"
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=pad_idx)
        self.pos_enc = _PositionalEncoding(embed_dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        pad_mask = (x == self.pad_idx)
        emb = self.pos_enc(self.embedding(x))
        out = self.transformer(emb, src_key_padding_mask=pad_mask)
        return self.fc(out[:, 0, :])


class TextCNNModel(nn.Module):
    def __init__(self, embedding_matrix, pad_idx: int = 0,
                 num_filters: int = 128, kernel_sizes=(2, 3, 4),
                 num_classes: int = 2, dropout: float | None = None):
        super().__init__()
        if dropout is None:
            dropout = max(_seq_dropout(), 0.5)
        emb = torch.as_tensor(embedding_matrix)
        embed_dim = emb.shape[1]
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))
            p = c.max(dim=2).values
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))


SEQ_BUILDERS = {
    "MeanPooling": MeanPoolingModel,
    "LSTM": LSTMModel,
    "Transformer": TransformerModel,
    "TextCNN": TextCNNModel,
}
