import math
from typing import Tuple

import torch
import torch.nn as nn

from src.algo.boulderdash import BoulderDash
from src.models.util import *


class ResConvLayer(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            act: str = "gelu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, padding=int(math.floor(kernel_size / 2))
        )
        self.act = activation_to_module(act)

    def forward(self, x):
        y = self.conv(x)
        if self.act is not None:
            y = self.act(y)
        return y + x


class BoulderDashPredictModel(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int],
            num_hidden: int,
            num_layers: int = 1,
            kernel_size: int = 3,
            act: str = "gelu",
    ):
        super().__init__()

        self._input_shape = (BoulderDash.OBJECTS.count() + BoulderDash.STATES.count(), *shape)
        padding = int(math.floor(kernel_size / 2))
        self.layers = nn.Sequential(
            nn.Conv2d(self._input_shape[0], num_hidden, kernel_size=kernel_size, padding=padding),
        )
        if act is not None:
            self.layers.append(activation_to_module(act))

        for i in range(num_layers):
            self.layers.append(ResConvLayer(num_hidden, kernel_size=kernel_size, act=act))

        self.layers.append(
            nn.Conv2d(num_hidden, self._input_shape[0], kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        #num_obj = BoulderDash.OBJECTS.count()
        #x[..., :num_obj, :, :] = F.softmax(x[..., :num_obj, :, :], -3)
        #x[..., num_obj:, :, :] = F.softmax(x[..., num_obj:, :, :], -3)
        return x




