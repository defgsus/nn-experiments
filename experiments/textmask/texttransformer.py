from typing import Optional, Union, Type, Callable, Iterable

import torch
import torch.nn as nn

from src.models.encoder import DiagonalEmbedding


class TextTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_channels: int,
            num_channels_mlp: int,
            num_heads: int,
            activation: Union[None, str, Callable] = "relu",
            diagonal_embedding: bool = True,
            symmetric_embedding: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = DiagonalEmbedding(
            channels_in=vocab_size,
            channels_out=num_channels,
            diagonal=diagonal_embedding,
            symmetric=symmetric_embedding,
        )
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=num_channels,
                nhead=num_heads,
                dim_feedforward=num_channels_mlp,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, logits: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(logits)
        x = x.permute(0, 2, 1)

        x = self.transformer(x, x)

        x = x.permute(0, 2, 1)
        return self.embedding(x, reverse=True)
