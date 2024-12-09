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


class Attention(nn.Module):
    def __init__(
            self,
            channels: int,
            heads: int,
            activation: Union[None, str, Callable] = None,
            dropout: float = 0.,
            residual: bool = True,
    ):
        super().__init__()
        self._is_residual = residual
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
        )
        self.act = activation_to_module(activation)

    def extra_repr(self) -> str:
        return f"residual={self._is_residual}"

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim in (2, 3):
            x_flat = x
            x_reverse = None
        elif x.ndim == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).view(B, H * W, C)
            x_reverse = lambda x: x.view(B, H, W, C).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError(f"ndim must be 2, 3, 4, got {x.shape}")

        if state is not None:
            assert x.shape == state.shape, f"Expected {x.shape}, got {state.shape}"
            if state.ndim in (2, 3):
                state_flat = state
            elif x.ndim == 4:
                B, C, H, W = state.shape
                state_flat = x.permute(0, 2, 3, 1).view(B, H * W, C)
            else:
                raise NotImplementedError(f"ndim must be 2, 3, 4, got {x.shape}")

            y = self.attn(x_flat, state_flat, x_flat)[0]
        else:
            y = self.attn(x_flat, x_flat, x_flat)[0]

        if self.act is not None:
            y = self.act(y)

        if x_reverse is not None:
            y = x_reverse(y)

        if self._is_residual:
            y = y + x
        return y


class ConvMixer(nn.Module):

    def __init__(
            self,
            channels: int,
            factor: int = 8,
            num_layers: int = 1,
            kernel_size: int = 5,
            activation: Union[None, str, Callable] = None,
            batch_norm: bool = False,
    ):
        super().__init__()

        padding = int(math.floor(kernel_size / 2))
        chan_mult = factor ** 2
        cur_channels = channels

        self.encoder = nn.Sequential()
        for i in range(num_layers):
            self.encoder.add_module(f"unshuffle_{i + 1}", nn.PixelUnshuffle(factor))
            cur_channels = cur_channels * chan_mult
            self.encoder.add_module(f"conv_{i + 1}", nn.Conv2d(cur_channels, cur_channels, kernel_size, padding=padding, groups=cur_channels))
            if activation is not None:
                self.encoder.add_module(f"act_{i + 1}", activation_to_module(activation))
            if batch_norm and i < num_layers - 1:
                self.encoder.add_module(f"norm_{i + 1}", nn.BatchNorm2d(cur_channels))

        self.decoder = nn.Sequential()
        for i in range(num_layers):
            self.decoder.add_module(f"shuffle_{i + 1}", nn.PixelShuffle(factor))
            cur_channels = cur_channels // chan_mult
            self.decoder.add_module(f"conv_{i + 1}", nn.Conv2d(cur_channels, cur_channels, kernel_size, padding=padding, groups=cur_channels))
            if activation is not None and i < num_layers - 1:
                self.decoder.add_module(f"act_{i + 1}", activation_to_module(activation))
            if batch_norm and i < num_layers - 1:
                self.decoder.add_module(f"norm_{i + 1}", nn.BatchNorm2d(cur_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)) + x


class ConvStrideMixer(nn.Module):

    def __init__(
            self,
            channels: int,
            stride: int,
            hidden_channels: Optional[int] = None,
            activation: Union[None, str, Callable] = None,
            residual: bool = True,
    ):
        super().__init__()
        self._channels = channels
        self._hidden_channels = channels * stride if hidden_channels is None else hidden_channels
        self._stride = stride
        self._residual = residual

        self.encoder = nn.Sequential()
        self.encoder.add_module("patch", nn.Conv2d(channels, self._hidden_channels, stride * 2, stride=stride))
        if activation is not None:
            self.encoder.add_module("act", activation_to_module(activation))
        self.encoder.add_module("conv", nn.Conv2d(self._hidden_channels, self._hidden_channels, 1))
        self.encoder.add_module("unpatch", nn.ConvTranspose2d(self._hidden_channels, channels, stride * 2, stride=stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        if self._residual:
            y = y + x
        return y

    def extra_repr(self):
        return f"channels={self._channels}, stride={self._stride}, residual={self._residual}"


class Layer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            batch_norm: bool = True,
            batch_norm_pos: int = 0,
            activation: Union[None, str, Callable] = "gelu",
            padding_mode: str = "zeros",
            init_weights: Optional[float] = None,
            attention: bool = False,
            state_channels: int = 0,
            gating: bool = False,
            gating_activation: Union[None, str, Callable] = "sigmoid",
    ):
        super().__init__()
        self._batch_norm_pos = batch_norm_pos
        self._state_channels = state_channels

        true_in_channels = in_channels + state_channels if not attention else in_channels

        if batch_norm and batch_norm_pos == 0:
            self.bn = nn.BatchNorm2d(true_in_channels)

        self.attention = None
        if attention:
            self.attention = Attention(
                channels=in_channels,
                heads=4,
                activation=activation,
            )

        self.conv = nn.Conv2d(
            in_channels=true_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=int(math.floor(kernel_size / 2)),
            padding_mode=padding_mode,
        )

        #self.gating_mixer = ConvMixer(out_channels, activation=activation, batch_norm=batch_norm)
        #self.gating_mixer = ConvStrideMixer(out_channels, stride=16, activation=activation, hidden_channels=out_channels)
        #self.gating_act = activation_to_module(gating_activation)

        if batch_norm and batch_norm_pos == 1:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = activation_to_module(activation)

        if batch_norm and batch_norm_pos == 2:
            self.bn = nn.BatchNorm2d(out_channels)

        if init_weights is not None:
            with torch.no_grad():
                for p in self.parameters():
                    p[:] = p * init_weights

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_x = x

        if self._state_channels > 0:
            assert state is not None, f"state must be passed when state_channels == {self._state_channels}"

        if self.attention is None:
            x = torch.cat([x, state], dim=-3)

        if self._batch_norm_pos == 0 and hasattr(self, "bn"):
            x = self.bn(x)

        if self.attention is not None:
            x = self.attention(x, state)

        x = self.conv(x)

        if self._batch_norm_pos == 1 and hasattr(self, "bn"):
            x = self.bn(x)

        if self.act:
            x = self.act(x)

        if self._batch_norm_pos == 2 and hasattr(self, "bn"):
            x = self.bn(x)

        if original_x.shape == x.shape:
            x = x + original_x

        return x


class StackedModel(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            num_layers: Optional[int] = None,
            channels: Union[int, Iterable[int]] = 32,
            kernel_size: Union[int, Iterable[int]] = 3,
            padding_mode: str = "zeros",
            state_channels: int = 0,
            batch_norm: Union[bool, Iterable[bool]] = True,
            activation: Union[None, str, Callable] = "gelu",
            attention: Union[int, Iterable[int]] = 0,
            init_weights: Optional[float] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if num_layers is None:
            num_layers = 1

        channels = param_make_list(channels, num_layers, "channels") + [out_channels]
        kernel_sizes = param_make_list(kernel_size, num_layers, "kernel_size") + [3]
        batch_norms = param_make_list(batch_norm, num_layers, "batch_norm") + [False]
        attentions = [0] + param_make_list(attention, num_layers-1, "attention") + [0]

        channels_list = [in_channels, *channels]

        self.state_conv = None
        if state_channels > 0:
            self.state_conv = nn.Conv2d(in_channels, state_channels, 3, padding=1)

        self.layers = nn.ModuleDict()

        with torch.no_grad():

            for i, (ch, ch_next, kernel_size, batch_norm, attn) in enumerate(
                    zip(channels_list, channels_list[1:], kernel_sizes, batch_norms, attentions)
            ):
                self.layers[f"layer_{i + 1}"] = Layer(
                    in_channels=ch,
                    out_channels=ch_next,
                    kernel_size=kernel_size,
                    batch_norm=batch_norm and i < num_layers - 1,
                    activation=(activation if i < num_layers - 1 else None),
                    padding_mode=padding_mode,
                    init_weights=init_weights,
                    state_channels=state_channels,
                    attention=attn,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        state = x

        if self.state_conv is not None:
            state = self.state_conv(state)

        for layer in self.layers.values():
            y = layer(y, state)

        return y

