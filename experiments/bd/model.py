import math
import dataclasses
from typing import Tuple

import torch
import torch.nn as nn

from src.algo.boulderdash import BoulderDash
from src.models.util import *


@dataclasses.dataclass
class BDModelInput:
    # [N, OBJS + STATES, H, W]
    state: torch.Tensor
    # [N, ACTIONS]
    action: Optional[torch.Tensor] = None


@dataclasses.dataclass
class BDModelOutput:
    next_state: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None
    action: Optional[torch.Tensor] = None


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


class ResConvLayers(nn.Module):

    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            channels_hidden: int,
            num_layers: int = 1,
            kernel_size: int = 3,
            act: str = "gelu",
    ):
        super().__init__()

        padding = int(math.floor(kernel_size / 2))
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_hidden, kernel_size=kernel_size, padding=padding),
        )
        if act is not None:
            self.layers.append(activation_to_module(act))

        for i in range(num_layers):
            self.layers.append(ResConvLayer(channels_hidden, kernel_size=kernel_size, act=act))

        self.layers.append(
            nn.Conv2d(channels_hidden, channels_out, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

        #num_obj = BoulderDash.OBJECTS.count()
        #x[..., :num_obj, :, :] = F.softmax(x[..., :num_obj, :, :], -3)
        #x[..., num_obj:, :, :] = F.softmax(x[..., num_obj:, :, :], -3)


class BoulderDashActionPredictModel(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int],
            num_hidden: int,
            num_layers: int = 1,
            kernel_size: int = 3,
            act: str = "gelu",
    ):
        super().__init__()

        self._num_actions = BoulderDash.ACTIONS.count()
        self._num_states = BoulderDash.OBJECTS.count() + BoulderDash.STATES.count()
        self.predict_state = ResConvLayers(
            channels_in=self._num_states + self._num_actions,
            channels_out=self._num_states,
            channels_hidden=num_hidden,
            num_layers=num_layers,
            kernel_size=kernel_size,
            act=act,
        )
        self.predict_reward = nn.Sequential(
            ResConvLayers(
                channels_in=self._num_states * 2 + self._num_actions,
                channels_out=num_hidden,
                channels_hidden=num_hidden,
                num_layers=num_layers,
                kernel_size=kernel_size,
                act=act,
            ),
            nn.Flatten(-3),
            nn.Linear(shape[0] * shape[1] * num_hidden, shape[0] * shape[1] * num_hidden // 2),
            activation_to_module(act),
            nn.Linear(shape[0] * shape[1] * num_hidden // 2, 1),
        )

    def forward(self, input: BDModelInput) -> BDModelOutput:
        action_map = input.action[:, :, None, None].repeat(1, 1, *input.state.shape[-2:])
        state_and_action = torch.concat([input.state, action_map], -3)

        next_state = self.predict_state(state_and_action)
        reward = self.predict_reward(
            torch.concat([state_and_action, next_state], -3)
        )
        return BDModelOutput(
            next_state=next_state,
            reward=reward,
        )
