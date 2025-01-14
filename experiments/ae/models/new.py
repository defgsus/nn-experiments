import unittest
import random
import math
import time
import warnings
from io import BytesIO
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union

import PIL.Image
import PIL.ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.datasets import *
from src.util.image import *
from src.util import *
from src.algo import *
from src.models.decoder import *
from src.models.transform import *
from src.models.util import *


class ConvLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            shuffle: int = 0,
            shuffle_compensate: bool = True,
            batch_norm: bool = False,
            activation: Union[None, str, Callable] = "relu6",
            transposed: bool = False,
    ):
        super().__init__()
        self._transposed = transposed
        self._shuffle = shuffle
        self._shuffle_compensate = shuffle_compensate

        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(in_channels)

        conv_class = nn.Conv2d

        if transposed:
            conv_class = nn.ConvTranspose2d

        if self._shuffle > 1 and self._transposed:
            if self._shuffle_compensate:
                self.shuffle_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * (self._shuffle ** 2),
                    kernel_size=1,
                )
            self.shuffle = nn.PixelShuffle(self._shuffle)

        self.conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if self._shuffle > 1 and not self._transposed:
            self.unshuffle = nn.PixelUnshuffle(self._shuffle)
            if self._shuffle_compensate:
                self.unshuffle_conv = nn.Conv2d(
                    in_channels=out_channels * (self._shuffle ** 2),
                    out_channels=out_channels,
                    kernel_size=1,
                )

        self.act = activation_to_module(activation)

    def extra_repr(self) -> str:
        return f"shuffle={self._shuffle}"

    def forward(
            self,
            x: torch.Tensor,
            output_size: Union[None, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        original_x = x

        if self.bn is not None:
            x = self.bn(x)

        if self._shuffle > 1 and self._transposed:
            if self._shuffle_compensate:
                x = self.shuffle_conv(x)
                if self.act is not None:
                    x = self.act(x)
            x = self.shuffle(x)

        x = self.conv(x)

        if output_size is not None and tuple(x.shape[-2:]) != output_size:
            x = F.pad(x, (0, output_size[-1] - x.shape[-1], 0, output_size[-2] - x.shape[-2]))

        if self.act:
            x = self.act(x)

        if self._shuffle and not self._transposed:
            x = self.unshuffle(x)
            if self._shuffle_compensate:
                x = self.unshuffle_conv(x)
                if self.act:
                    x = self.act(x)

        if x.shape == original_x.shape:
            x = x + original_x

        return x


class NewAutoEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            num_layers: int,
            out_channels: Optional[int] = None,
            channels: Union[int, Iterable[int]] = 32,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: Union[int, Iterable[int]] = 1,
            padding: Union[int, Iterable[int]] = 0,
            dilation: Union[int, Iterable[int]] = 1,
            shuffle: Union[int, Iterable[int]] = 0,
            batch_norm: Union[bool, Iterable[bool]] = True,
            activation: Union[None, str, Callable] = "relu6",
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        channels_ = param_make_list(channels, num_layers, "channels")
        kernel_size_ = param_make_list(kernel_size, num_layers, "kernel_size")
        stride_ = param_make_list(stride, num_layers, "stride")
        padding_ = param_make_list(padding, num_layers, "padding")
        dilation_ = param_make_list(dilation, num_layers, "dilation")
        shuffle_ = param_make_list(shuffle, num_layers, "shuffle")
        batch_norm_ = param_make_list(batch_norm, num_layers, "batch_norm")

        channels_list = [in_channels, *channels_]

        self.encoder = nn.Sequential()

        ch_mult = 1
        used_channels = []
        for i, (ch, ch_next, ks, stride, pad, dilation, shuffle, bn) in enumerate(
                zip(channels_list, channels_list[1:], kernel_size_, stride_, padding_, dilation_, shuffle_, batch_norm_)
        ):
            is_last_layer = i == num_layers - 1

            self.encoder.add_module(f"layer_{i + 1}", ConvLayer(
                in_channels=ch * ch_mult,
                out_channels=ch_next if is_last_layer else (ch_next * ch_mult),
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                shuffle=shuffle,
                shuffle_compensate=False,
                batch_norm=None if is_last_layer else batch_norm,
                activation=None if is_last_layer else activation,
            ))
            used_channels.append(ch * ch_mult)
            if is_last_layer:
                used_channels.append(ch_next)
            if shuffle > 1:
                ch_mult = shuffle ** 2

        #channels_list = list(reversed([out_channels, *channels_]))
        channels_list = list(reversed(used_channels))
        kernel_size_ = list(reversed(kernel_size_))
        stride_ = list(reversed(stride_))
        padding_ = list(reversed(padding_))
        dilation_ = list(reversed(dilation_))
        shuffle_ = list(reversed(shuffle_))
        batch_norm_ = list(reversed(batch_norm_))

        self.decoder = nn.Sequential()
        for i, (ch, ch_next, ks, stride, pad, dilation, shuffle, bn) in enumerate(
                zip(channels_list, channels_list[1:], kernel_size_, stride_, padding_, dilation_, shuffle_, batch_norm_)
        ):
            is_last_layer = i == num_layers - 1

            self.decoder.add_module(f"layer_{i + 1}", ConvLayer(
                in_channels=ch,
                out_channels=ch_next,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                shuffle=shuffle,
                shuffle_compensate=False,
                batch_norm=None if is_last_layer else batch_norm,
                activation=None if is_last_layer else activation,
                transposed=True,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)

        y = self.decoder(y)

        return y
