import unittest
import random
import math
import time
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
            batch_norm: bool = True,
            batch_norm_pos: int = 0,
            activation: Union[None, str, Callable] = "gelu",
            padding_mode: str = "zeros",
            transposed: bool = False,
    ):
        super().__init__()
        self._batch_norm_pos = batch_norm_pos

        if batch_norm and batch_norm_pos == 0:
            self.bn = nn.BatchNorm2d(in_channels)

        self.conv = (nn.ConvTranspose2d if transposed else nn.Conv2d)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )

        if batch_norm and batch_norm_pos == 1:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = activation_to_module(activation)

        if batch_norm and batch_norm_pos == 2:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(
            self,
            x: torch.Tensor,
            output_size: Union[None, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self._batch_norm_pos == 0 and hasattr(self, "bn"):
            x = self.bn(x)

        x = self.conv(x)

        if output_size is not None and tuple(x.shape[-2:]) != output_size:
            x = F.pad(x, (0, output_size[-1] - x.shape[-1], 0, output_size[-2] - x.shape[-2]))

        if self._batch_norm_pos == 1 and hasattr(self, "bn"):
            x = self.bn(x)

        if self.act:
            x = self.act(x)

        if self._batch_norm_pos == 2 and hasattr(self, "bn"):
            x = self.bn(x)

        return x


class ResConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            num_layers: Optional[int] = None,
            channels: Union[int, Iterable[int]] = 32,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: Union[int, Iterable[int]] = 1,
            padding: Union[int, Iterable[int]] = 0,
            padding_mode: str = "zeros",
            batch_norm: Union[bool, Iterable[bool]] = True,
            activation: Union[None, str, Callable] = "gelu",
            activation_last_layer: Union[None, str, Callable] = None,
            residual_weight: float = 1.,
            batch_norm_pos_encoder: int = 0,
            batch_norm_pos_decoder: int = 0,
    ):
        super().__init__()
        self.residual_weight = residual_weight

        if out_channels is None:
            out_channels = in_channels

        if num_layers is None:
            num_layers = 1

        channels = param_make_list(channels, num_layers, "channels")
        kernel_sizes = param_make_list(kernel_size, num_layers, "kernel_size")
        strides = param_make_list(stride, num_layers, "stride")
        paddings = param_make_list(padding, num_layers, "padding")
        batch_norms = param_make_list(batch_norm, num_layers, "batch_norm")

        channels_list = [in_channels, *channels]

        self.encoder = nn.ModuleDict()

        with torch.no_grad():

            for i, (ch, ch_next, kernel_size, stride, pad, batch_norm) in enumerate(
                    zip(channels_list, channels_list[1:], kernel_sizes, strides, paddings, batch_norms)
            ):
                self.encoder[f"layer_{i + 1}"] = ConvLayer(
                    in_channels=ch,
                    out_channels=ch_next,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    batch_norm=batch_norm and i < num_layers - 1,
                    activation=(activation if i < num_layers - 1 else None),
                    padding_mode=padding_mode,
                    batch_norm_pos=batch_norm_pos_encoder,
                )

        channels_list = list(reversed([out_channels, *channels]))
        kernel_sizes = list(reversed(kernel_sizes))
        strides = list(reversed(strides))
        paddings = list(reversed(paddings))
        batch_norms = list(reversed(batch_norms))

        self.decoder = nn.ModuleDict()
        for i, (ch, ch_next, kernel_size, stride, pad, batch_norm) in enumerate(
                zip(channels_list, channels_list[1:], kernel_sizes, strides, paddings, batch_norms)
        ):
            self.decoder[f"layer_{i + 1}"] = ConvLayer(
                in_channels=ch,
                out_channels=ch_next,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                batch_norm=batch_norm and i < num_layers - 1,
                activation=activation if i < num_layers - 1 else activation_last_layer,
                padding_mode=padding_mode,
                transposed=True,
                batch_norm_pos=batch_norm_pos_decoder,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        state_history = [state]
        for layer in self.encoder.values():
            state = layer(state)
            state_history.append(state)

        for i, layer in enumerate(self.decoder.values()):
            if i > 0:
                state = state + self.residual_weight * state_history[-(i + 1)]

            state = layer(state, output_size=state_history[-(i + 2)].shape[-2:])

        return state


class TestResConv(unittest.TestCase):

    @torch.no_grad()
    def test_resconv(self):
        for params in iter_parameter_permutations({
            "kernel_size": [1, 2, 3],
            "stride": [1, 2, 3],
            "padding": [0, 1, 2],
            "shape": [
                (3, 32, 32),
                (1, 31, 33),
            ]
        }):
            msg = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
            shape = params["shape"]

            model = ResConv(
                in_channels=shape[0],
                num_layers=3,
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
                activation_last_layer="sigmoid",
            ).eval()
            input = torch.rand(1, *shape)
            output = model(input)

            self.assertEqual(
                input.shape,
                output.shape,
                msg,
            )

        print(model)
