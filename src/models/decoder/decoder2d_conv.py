import math
from collections import OrderedDict
from typing import List, Iterable, Tuple, Optional, Callable, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.models.cnn import Conv2dBlock
from src.util.image import set_image_channels, image_resize_crop
from src.util import to_torch_device
from src.models.util import activation_to_module


class DecoderConv2d(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: int = 1,
            channels: Iterable[int] = (16, 32),
            activation: Union[None, str, Callable, nn.Module, Type[nn.Module]] = "relu",
            activation_last_layer: Union[None, bool, str, Callable, nn.Module, Type[nn.Module]] = None,
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.shape = shape
        self.code_size = code_size
        act_fn = activation_to_module(activation)
        # self.act_fn = act_fn

        channels = [*self.channels, self.shape[0]]
        self.convolution = Conv2dBlock(
            channels=channels,
            kernel_size=self.kernel_size,
            act_fn=act_fn,
            stride=self.stride,
            transpose=True,
            act_last_layer=activation_last_layer,
        )
        self.encoded_shape = self.convolution.get_input_shape(shape)
        self.linear = nn.Linear(self.code_size, math.prod(self.encoded_shape))

    @property
    def device(self):
        return self.linear.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolution(self.linear(x).view(-1, *self.encoded_shape))

    def get_extra_state(self):
        return {
            # **super().get_extra_state(),
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
