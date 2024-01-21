"""
Yeah well, this is complete nonsense or say, the idea didn't work in several ways
"""
import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module


class AlmostResidual2d(nn.Module):

    def __init__(self, kernel_size: int, transpose: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.transpose = transpose

    def forward(self, x):
        if self.kernel_size <= 1:
            return x

        num = self.kernel_size // 2
        if not self.transpose:
            return x[..., num:-num, num:-num]

        else:
            return F.pad(x, (num, num, num, num))

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, transpose={self.transpose}"



class ResConvBlock(nn.Module):

    def __init__(
            self,
            channels_in: int,
            channels_hidden: int,
            channels_out: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            activation: Union[None, str, Callable, nn.Module, Type[nn.Module]] = "relu",
            batch_norm: bool = True,
            residual: Union[bool, str] = "add",
            transpose: bool = False,
    ):
        assert residual in (False, True, "add", "concat"), f"Got: {residual}"
        if residual is True:
            residual = "add"
        if residual == "add" and channels_in != channels_out:
            raise ValueError(
                f"channels_in/_out must be equal for residual='add' mode"
            )

        super().__init__()

        conv_class = nn.Conv2d if not transpose else nn.ConvTranspose2d

        self.residual = residual
        self.res_layers = nn.Sequential()
        if residual:
            if kernel_size > 1:
                self.res_layers.add_module(
                    f"pool{kernel_size}x{kernel_size}",
                    #nn.AvgPool2d(kernel_size=kernel_size, stride=1)
                    AlmostResidual2d(kernel_size=kernel_size, transpose=transpose)
                )

        self.layers = nn.Sequential()
        self.layers.add_module("c1x1_1", conv_class(channels_in, channels_hidden, kernel_size=1, groups=groups))
        if batch_norm:
            self.layers.add_module("bn_1", nn.BatchNorm2d(channels_hidden))
        if activation is not None:
            self.layers.add_module("act_1", activation_to_module(activation))

        self.layers.add_module(f"c{kernel_size}x{kernel_size}", conv_class(channels_hidden, channels_hidden, kernel_size=kernel_size, groups=groups))
        if batch_norm:
            self.layers.add_module("bn_2", nn.BatchNorm2d(channels_hidden))
        if activation is not None:
            self.layers.add_module("act_2", activation_to_module(activation))
        self.layers.add_module("c1x1_2", conv_class(channels_hidden, channels_out, kernel_size=1, groups=groups))
        if batch_norm:
            self.layers.add_module("bn_3", nn.BatchNorm2d(channels_out))

        if activation is not None:
            self.act_3 = activation_to_module(activation)

    def forward(self, x):
        y = self.layers(x)

        x_res = self.res_layers(x)
        #print("y:", y.shape, "x_res:", x_res.shape)

        if self.residual == "add":
            y = x_res + y
        elif self.residual == "concat":
            # print(x_res.shape, y.shape)
            y = torch.concat([x_res, y], dim=-3)
        return y


class ResConv2dDecoder(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: int = 1,
            groups: int = 1,
            channels: Iterable[int] = (256, 256, 256, 256, 128, 128, 64, 64, 32, 32, 3),
            activation: Union[None, str, Callable, nn.Module, Type[nn.Module]] = "relu",
            activation_last_layer: Union[None, bool, str, Callable, nn.Module, Type[nn.Module]] = None,
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.shape = shape
        self.code_size = code_size
        #act_fn = activation_to_module(activation)

        conv = nn.Sequential()
        ch_add = 0
        for i, (ch, ch_next) in enumerate(zip(self.channels, self.channels[1:])):
            is_last_layer = i + 2 == len(self.channels)
            conv.add_module(
                f"block{i+1}",
                ResConvBlock(
                    channels_in=ch + ch_add,
                    channels_hidden=ch * 2,
                    channels_out=ch_next,
                    kernel_size=kernel_size,
                    groups=1 if is_last_layer else groups,
                    transpose=True,
                    activation=activation,
                    residual=False if is_last_layer else "concat",
                )
            )
            ch_add += ch

        self.conv_shape = (self.channels[0], 24, 24)

        self.linear = nn.Linear(code_size, math.prod(self.conv_shape))
        self.conv = conv

    def forward(self, x):
        bs = x.shape[0]
        y = self.linear(x).view(bs, *self.conv_shape)
        return self.conv(y)
