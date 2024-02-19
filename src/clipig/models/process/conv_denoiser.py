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


class ConvDenoiser(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int],
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: Union[int, Iterable[int]] = 1,
            batch_norm: bool = True,
            activation: Union[None, str, Callable] = "gelu",
    ):
        super().__init__()
        self.shape = shape

        self._channels = [shape[0], *channels, shape[0]]
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size] * len(self._channels)
        else:
            kernel_sizes = list(kernel_size)
            if len(kernel_sizes) != len(self._channels) - 1:
                raise ValueError(f"Expected kernel_size of length {len(self._channels) - 1}, got {len(kernel_sizes)}")

        if isinstance(stride, int):
            strides = [stride] * len(self._channels)
        else:
            strides = list(stride)
            if len(strides) != len(self._channels) - 1:
                raise ValueError(f"Expected stride of length {len(self._channels) - 1}, got {len(strides)}")

        self.encoder = nn.ModuleDict()
        decoder_paddings = []
        with torch.no_grad():
            tmp_state = torch.zeros(1, *shape)

            for i, (ch, ch_next, kernel_size, stride) in enumerate(zip(self._channels, self._channels[1:], kernel_sizes, strides)):
                if batch_norm:
                    self.encoder[f"layer{i+1}_bn"] = nn.BatchNorm2d(ch)
                self.encoder[f"layer{i+1}_conv"] = nn.Conv2d(ch, ch_next, kernel_size, stride=stride)
                if activation:
                    self.encoder[f"layer{i+1}_act"] = activation_to_module(activation)

                in_shape = tmp_state.shape[-2:]
                tmp_state = self.encoder[f"layer{i+1}_conv"](tmp_state)
                dec_shape = nn.ConvTranspose2d(ch_next, ch_next, kernel_size, stride=stride)(tmp_state).shape[-2:]
                decoder_paddings.append(
                    [s - ds for s, ds in zip(in_shape, dec_shape)]
                )

        channels = list(reversed(self._channels))
        kernel_sizes = list(reversed(kernel_sizes))
        strides = list(reversed(strides))
        self.decoder = nn.ModuleDict()
        for i, (ch, ch_next, kernel_size, stride, pad) in enumerate(
                zip(channels, channels[1:], kernel_sizes, strides, list(reversed(decoder_paddings)))
        ):
            if batch_norm:
                self.decoder[f"layer{i+1}_bn"] = nn.BatchNorm2d(ch)

            self.decoder[f"layer{i+1}_conv"] = nn.ConvTranspose2d(
                ch, ch_next, kernel_size, stride=stride, output_padding=pad,
            )
            if activation and i < len(channels) - 2:
                self.decoder[f"layer{i+1}_act"] = activation_to_module(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_history = []
        state = x
        for i in range(len(self._channels) - 1):
            if f"layer{i+1}_bn" in self.encoder:
                state = self.encoder[f"layer{i+1}_bn"](state)
            state = self.encoder[f"layer{i+1}_conv"](state)
            if f"layer{i+1}_act" in self.encoder:
                state = self.encoder[f"layer{i+1}_act"](state)
            state_history.append(state)
            #print("ENC", state.shape)

        for i in range(len(self._channels) - 1):
            if i > 0:
                # print("DEC", state.shape, state_history[-(i+1)].shape)
                state = state + state_history[-(i+1)]
            if f"layer{i+1}_bn" in self.decoder:
                state = self.decoder[f"layer{i+1}_bn"](state)
            state = self.decoder[f"layer{i+1}_conv"](state)
            if f"layer{i+1}_act" in self.decoder:
                state = self.decoder[f"layer{i+1}_act"](state)

        return state



class TestConvDenoiser(unittest.TestCase):

    @torch.no_grad()
    def test_denoiser(self):
        for shape in (
                (3, 64, 64),
                (3, 63, 65),
        ):
            for stride in (
                    (1, 1),
                    (1, 1, 1),
                    (1, 1, 1, 1),
                    (1, 1, 1, 1, 1),
                    (1, 2, 2, 2, 1),
                    (1, 2, 2, 2, 1),
                    (2, 2, 2, 2, 2),
                    (2, 3, 4),
            ):
                msg = f"shape={shape}, stride={stride}"

                model = ConvDenoiser(
                    shape=shape,
                    channels=[16] * (len(stride) - 1),
                    stride=stride,
                ).eval()

                self.assertEqual(
                    torch.Size(shape),
                    model(torch.zeros(1, *shape)).shape[-3:],
                    msg
                )
