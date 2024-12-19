import math
from typing import Union, Callable, Type, Dict, List, Iterable

import torch
import torch.nn as nn

from src.models.util import activation_to_module, normalization_to_module
from src.util.params import param_make_tuple



class LstmTextModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_channels: Union[int, Iterable[int]],
            dropout: float = 0.,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, num_channels)
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=num_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(num_channels, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        # x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        # x = x.permute(0, 2, 1)

        logits = self.lm_head(x)

        return logits
