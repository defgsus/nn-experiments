import math
from collections import OrderedDict
from typing import List, Iterable, Tuple, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.models.cnn import Conv2dBlock
from src.util.image import set_image_channels, image_resize_crop
from src.util import to_torch_device
from .base2d import Encoder2d


class EncoderConv2d(Encoder2d):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: int = 1,
            channels: Iterable[int] = (16, 32),
            act_fn: Union[None, str, nn.Module] = nn.ReLU(),
            space_to_depth: bool = False,
            dropout: float = 0.,
            batch_norm: bool = False,
    ):
        super().__init__(shape=shape, code_size=code_size)
        self.channels = tuple(channels)
        self.kernel_size = kernel_size
        self.stride = stride
        # self.act_fn = act_fn

        channels = [self.shape[0], *self.channels]
        self.convolution = Conv2dBlock(
            channels=channels,
            kernel_size=self.kernel_size,
            act_fn=act_fn,
            stride=self.stride,
            space_to_depth=space_to_depth,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        encoded_shape = self.convolution.get_output_shape(shape)
        self.linear = nn.Linear(math.prod(encoded_shape), self.code_size)

    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.convolution(x).flatten(1))

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "kernel_size": self.kernel_size,
            "channels": self.channels,
            "act_fn": self.convolution._act_fn,
        }

    @classmethod
    def from_data(cls, data: dict):
        extra = data["_extra_state"]
        model = cls(
            shape=extra["shape"],
            kernel_size=extra["kernel_size"],
            stride=extra.get("stride", 1),
            channels=extra["channels"],
            code_size=extra["code_size"],
            act_fn=extra["act_fn"],
        )
        model.load_state_dict(data)
        return model
