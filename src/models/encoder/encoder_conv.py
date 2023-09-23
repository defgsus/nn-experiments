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


class EncoderConv2d(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            kernel_size: int = 15,
            stride: int = 1,
            channels: Iterable[int] = (16, 32),
            code_size: int = 1024,
            act_fn: Optional[nn.Module] = nn.ReLU(),
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.channels = tuple(channels)
        self.kernel_size = int(kernel_size)
        self.code_size = int(code_size)
        self.stride = stride
        # self.act_fn = act_fn

        channels = [self.shape[0], *self.channels]
        self.convolution = Conv2dBlock(
            channels=channels,
            kernel_size=self.kernel_size,
            act_fn=act_fn,
            stride=self.stride,
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
            "shape": self.shape,
            "kernel_size": self.kernel_size,
            "channels": self.channels,
            "code_size": self.code_size,
            "act_fn": self.convolution._act_fn,
        }

    def set_extra_state(self, state):
        pass

    @classmethod
    def from_torch(cls, f, device: Union[None, str, torch.device] = "cpu"):
        if isinstance(f, (dict, OrderedDict)):
            data = f
        else:
            data = torch.load(f)

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
        return model.to(to_torch_device(device))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-3:] != self.shape:
            image = image_resize_crop(image, self.shape[-2:])
            image = set_image_channels(image, self.shape[-3])

        return self(image)
