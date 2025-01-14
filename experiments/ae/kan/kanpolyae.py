import math
from typing import Tuple, List, Union, Callable

import torch
import torch.nn as nn

from src.models.kan import KANPolyLayer
from src.models.transform import Reshape


class KANPolyAE(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: List[int],
            order: int,
            activation: Union[None, str, Callable] = None,
    ):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.encoder.append(nn.Flatten(1))
        channels = [math.prod(shape)] + list(channels)
        for ch, next_ch in zip(channels, channels[1:]):
            self.encoder.append(KANPolyLayer(
                ch, next_ch, order, activation=activation,
            ))
            self.decoder.insert(0, KANPolyLayer(
                next_ch, ch, order, activation=activation,
            ))
        self.decoder.append(Reshape(shape))

    def forward(self, x):
        return self.decoder(self.encoder(x))
