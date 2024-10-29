import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module
from src.util.params import param_make_tuple
from src.models.util import ResidualAdd


class RescaleConv(nn.Module):

    def __init__(
            self,
            factor: int,
            channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            activation: Union[None, str, Callable] = None,
            batch_norm: bool = False,
            transpose: bool = False,
    ):
        super().__init__()

        self._transpose = transpose
        chan_mult = factor ** 2
        if not transpose:
            self.unshuffle = nn.PixelUnshuffle(factor)
            self.conv = nn.Conv2d(channels * chan_mult, channels, kernel_size, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(channels, channels * chan_mult, kernel_size, padding=padding)
            self.shuffle = nn.PixelShuffle(factor)

        self.act = activation_to_module(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._transpose:
            x = self.unshuffle(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.shuffle(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def extra_repr(self):
        return f"transpose={self._transpose}"


class ResConvStack(nn.Module):

    def __init__(
            self,
            channels: Tuple[int, ...],
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            padding: Union[int, Tuple[int, ...]] = 1,
            activation: Union[Union[None, str, Callable], Tuple[Union[None, str, Callable], ...]] = None,
            batch_norm: Union[bool, Tuple[bool, ...]] = False,
            depth: Union[int, Tuple[int, ...]] = 0,
            scale: Union[int, Tuple[int, ...]] = 1,
            transpose: bool = False,
    ):
        super().__init__()

        self._channels = tuple(channels)
        assert len(self._channels) >= 2, f"Got {len(self._channels)}"
        num_layers = len(self._channels) - 1
        self._kernel_size = param_make_tuple(kernel_size, num_layers, "kernel_size")
        self._padding = param_make_tuple(padding, num_layers, "padding")
        self._activation = param_make_tuple(activation, num_layers, "activation")
        self._batch_norm = param_make_tuple(batch_norm, num_layers, "batch_norm")
        self._depth = param_make_tuple(depth, num_layers, "depth")
        self._scale = param_make_tuple(scale, num_layers, "scale")
        self._transpose = transpose

        self.layers = nn.Sequential()
        for idx in range(num_layers):
            ch1 = self._channels[idx]
            ch2 = self._channels[idx + 1]

            if self._batch_norm[idx]:
                self.layers.add_module(f"layer_{idx+1}_norm", nn.BatchNorm2d(ch1))
            if self._scale[idx] > 1:
                self.layers.add_module(
                    f"layer_{idx+1}_scale",
                    RescaleConv(self._scale[idx], ch1, transpose=transpose)
                )

            conv_module = (nn.ConvTranspose2d if transpose else nn.Conv2d)(
                ch1, ch2, self._kernel_size[idx], padding=self._padding[idx]
            )
            act_module = activation_to_module(self._activation[idx])

            if False:
                if ch1 == ch2:
                    if self._activation is not None:
                        self.layers.add_module(
                            f"layer_{idx+1}_conv",
                            ResidualAdd(nn.Sequential(
                                conv_module,
                                act_module,
                            ))
                        )
                    else:
                        self.layers.add_module(f"layer_{idx+1}_conv", ResidualAdd(conv_module))
                else:
                    self.layers.add_module(f"layer_{idx+1}_conv", conv_module)
                    if self._activation[idx] is not None:
                        self.layers.add_module(f"layer_{idx+1}_act", act_module)
            else:
                if ch1 == ch2:
                    conv_module = ResidualAdd(conv_module)
                self.layers.add_module(f"layer_{idx+1}_conv", conv_module)
                if self._activation[idx] is not None:
                    self.layers.add_module(f"layer_{idx+1}_act", act_module)

            for i in range(self._depth[idx]):
                self.layers.add_module(
                    f"layer_{idx+1}_res_{i+1}",
                    ResidualAdd(
                        nn.Conv2d(ch2, ch2, self._kernel_size[idx], padding=int(math.floor(self._kernel_size[idx] / 2)))
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def extra_repr(self):
        return (
            f"channels={self._channels}, depth={self._depth}, scale={self._scale}, kernel_size={self._kernel_size}"
            f",\nbatch_norm={self._batch_norm}, activation={self._activation}, "
        )
