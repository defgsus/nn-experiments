import math
from typing import Tuple, List, Union, Callable

import torch
import torch.nn as nn

from src.models.transform import Reshape
from src.models.util import activation_to_module


class MLPAE(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: List[int],
            activation: Union[None, str, Callable] = None,
    ):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.encoder.append(nn.Flatten(1))
        channels = [math.prod(shape)] + list(channels)
        for ch, next_ch in zip(channels, channels[1:]):
            self.encoder.append(nn.Linear(ch, next_ch))
            if activation is not None:
                self.encoder.append(activation_to_module(activation))

            if activation is not None:
                self.decoder.insert(0, activation_to_module(activation))
            self.decoder.insert(0, nn.Linear(next_ch, ch))
        self.decoder.append(Reshape(shape))

    def forward(self, x):
        return self.decoder(self.encoder(x))
