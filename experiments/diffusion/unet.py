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
import torchvision.models
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


class UNetLayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            batch_norm: bool = True,
            batch_norm_pos: int = 0,
            activation: Union[None, str, Callable] = "gelu",
            padding_mode: str = "zeros",
            transposed: bool = False,
            embedding_size: int = 0,
    ):
        super().__init__()
        self._batch_norm_pos = batch_norm_pos
        self._embedding_size = embedding_size

        if batch_norm and batch_norm_pos == 0:
            self.bn = nn.BatchNorm2d(in_channels + embedding_size)

        self.conv = (nn.ConvTranspose2d if transposed else nn.Conv2d)(
            in_channels=in_channels + embedding_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
        )

        if batch_norm and batch_norm_pos == 1:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = activation_to_module(activation)

        if batch_norm and batch_norm_pos == 2:
            self.bn = nn.BatchNorm2d(out_channels)

    def extra_repr(self) -> str:
        params = []
        if self._embedding_size > 0:
            params.append(f"embedding_size={self._embedding_size}")
        return ", ".join(params)

    def forward(
            self,
            x: torch.Tensor,
            embedding: Optional[torch.Tensor] = None,
            output_size: Union[None, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        residual = x

        if self._embedding_size > 0:
            assert embedding is not None
            assert x.shape[:1] == embedding.shape[:1], f"Expected {x.shape}, got {embedding.shape}"
            assert embedding.shape[1] == self._embedding_size, f"Expected {self._embedding_size}, got {embedding.shape}"
            x = torch.cat([x, embedding], dim=1)

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

        if x.shape == residual.shape:
            x = x + residual

        return x


class UNet(nn.Module):

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
            conv_groups: Union[int, Iterable[int]] = 1,
            batch_norm: Union[bool, Iterable[bool]] = True,
            layer_embedding: Union[int, Iterable[int]] = 0,
            batch_norm_pos_encoder: int = 0,
            batch_norm_pos_decoder: int = 0,
            activation: Union[None, str, Callable] = "gelu",
            activation_last_layer: Union[None, str, Callable] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if num_layers is None:
            num_layers = 1

        channels = param_make_list(channels, num_layers, "channels")
        kernel_sizes = param_make_list(kernel_size, num_layers, "kernel_size")
        strides = param_make_list(stride, num_layers, "stride")
        paddings = param_make_list(padding, num_layers, "padding")
        batch_norms = param_make_list(batch_norm, num_layers, "batch_norm")
        conv_groups = param_make_list(conv_groups, num_layers, "conv_groups")
        self.layer_embedding = param_make_list(layer_embedding, num_layers, "layer_embedding")

        channels_list = [in_channels, *channels]

        self.encoder = nn.ModuleDict()

        with torch.no_grad():

            for i, (ch, ch_next, kernel_size, stride, pad, batch_norm, groups, emb_size) in enumerate(
                    zip(channels_list, channels_list[1:], kernel_sizes, strides, paddings, batch_norms, conv_groups, self.layer_embedding)
            ):
                self.encoder[f"layer_{i + 1}"] = UNetLayer(
                    in_channels=ch,
                    out_channels=ch_next,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    batch_norm=batch_norm and i < num_layers - 1,
                    activation=(activation if i < num_layers - 1 else None),
                    padding_mode=padding_mode,
                    batch_norm_pos=batch_norm_pos_encoder,
                    groups=groups,
                    embedding_size=emb_size,
                )

        channels_list = list(reversed([out_channels, *channels]))
        kernel_sizes = list(reversed(kernel_sizes))
        strides = list(reversed(strides))
        paddings = list(reversed(paddings))
        batch_norms = list(reversed(batch_norms))
        conv_groups = list(reversed(conv_groups))

        self.decoder = nn.ModuleDict()
        for i, (ch, ch_next, kernel_size, stride, pad, batch_norm, groups, emb_size) in enumerate(
                zip(channels_list, channels_list[1:], kernel_sizes, strides, paddings, batch_norms, conv_groups, self.layer_embedding)
        ):
            self.decoder[f"layer_{i + 1}"] = UNetLayer(
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
                groups=groups,
                embedding_size=emb_size,
            )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        state = x
        state_history = [state]
        for layer in self.encoder.values():
            state = layer(state, embedding)
            state_history.append(state)

        for i, layer in enumerate(self.decoder.values()):
            if i > 0:
                state = state + state_history[-(i + 1)]

            state = layer(state, embedding, output_size=state_history[-(i + 2)].shape[-2:])

        return state


torchvision.models.VisionTransformer