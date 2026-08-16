"""Small random backbone used only for deterministic pipeline smoke tests."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from torch import nn


class TinyTokenizer:
    """Stable whitespace tokenizer with no external files or network access."""

    def __init__(self, vocabulary_size: int = 2048) -> None:
        self.vocabulary_size = vocabulary_size

    def _id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return 2 + int.from_bytes(digest, "little") % (self.vocabulary_size - 2)

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        rows = [[self._id(token) for token in text.split()] for text in texts]
        max_length = max(map(len, rows))
        input_ids = torch.zeros((len(rows), max_length), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


class TinyCausalBackbone(nn.Module):
    """Qwen-compatible interface for checking multimodal tensor plumbing."""

    def __init__(self, hidden_size: int = 64, vocabulary_size: int = 2048) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size)
        )
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        del output_hidden_states, return_dict, use_cache
        length = inputs_embeds.shape[1]
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=inputs_embeds.device),
            diagonal=1,
        )
        hidden = self.encoder(
            inputs_embeds,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask.bool(),
        )
        return SimpleNamespace(hidden_states=(hidden,))
