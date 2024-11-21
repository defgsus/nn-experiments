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
            num_hidden: int = 16,
            num_layers: int = 6,
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
        return x


class BoulderDashPolicyModel(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int],
            prediction_model: BoulderDashPredictModel,
            num_hidden: Tuple[int, ...] = (256, 128),
            act: str = "gelu",
    ):
        super().__init__()
        self.prediction_model = prediction_model
        self.num_actions = BoulderDash.ACTIONS.count()
        self.map_shape = (BoulderDash.OBJECTS.count() + BoulderDash.STATES.count(), *shape)
        self.policy_layers = nn.Sequential()
        channels = [math.prod(self.map_shape) * 2, *num_hidden, self.num_actions]
        for i, ch1 in enumerate(channels[:-1]):
            ch2 = channels[i + 1]
            self.policy_layers.append(nn.Linear(ch1, ch2))
            if act is not None and i < len(channels) - 2:
                self.policy_layers.append(activation_to_module(act))

    def forward(
            self,
            state: torch.Tensor,
            return_prediction: bool = False,
    ) -> torch.Tensor:
        prediction = self.prediction_model(state)
        state = torch.cat([state, prediction], dim=-3)
        state = state.flatten(-3)
        action = self.policy_layers(state)
        if return_prediction:
            return action, prediction
        return action
