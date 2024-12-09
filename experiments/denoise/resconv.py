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


class SelfAttention(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim in (2, 3):
            x_flat = x
            x_reverse = None
        elif x.ndim == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).view(B, H * W, C)
            x_reverse = lambda x: x.view(B, H, W, C).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError(f"ndim must be 2, 3, 4, got {x.shape}")

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


class ConvDilationMixer(nn.Module):

    def __init__(
            self,
            channels: int,
            dilations: Tuple[int, ...] = (6, 5, 3),
            hidden_channels: Optional[int] = None,
            activation: Union[None, str, Callable] = None,
            residual: bool = True,
    ):
        super().__init__()
        self._channels = channels
        self._dilations = tuple(dilations)
        self._hidden_channels = channels if hidden_channels is None else hidden_channels
        self._residual = residual

        self.encoder = nn.Sequential()
        ch = channels
        next_ch = self._hidden_channels
        for i, dil in enumerate(dilations):
            self.encoder.add_module(f"conv_{i+1}", nn.Conv2d(ch, next_ch, 3, padding=dil, dilation=dil))
            if activation is not None:
                self.encoder.add_module(f"act_{i+1}", activation_to_module(activation))
            ch = next_ch
        self.encoder.add_module(f"conv_{i+2}", nn.Conv2d(ch, channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        if self._residual:
            y = y + x
        return y

    def extra_repr(self):
        return f"channels={self._channels}, dilations={self._dilations}, residual={self._residual}"


class ConvLayer(nn.Module):

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
            init_weights: Optional[float] = None,
            wavelet: bool = False,
            attention: bool = False,
            gating: bool = False,
            gating_activation: Union[None, str, Callable] = "sigmoid",
    ):
        super().__init__()
        self._batch_norm_pos = batch_norm_pos

        if batch_norm and batch_norm_pos == 0:
            self.bn = nn.BatchNorm2d(in_channels)

        self.attention = None
        if attention:
            self.attention = SelfAttention(
                channels=in_channels,
                heads=4,
                activation=activation,
            )

        conv_class = nn.Conv2d

        if wavelet and in_channels == out_channels:
            from src.models.wavelet import WTConv2d
            conv_class = WTConv2d
            if padding or groups:
                warnings.warn("Can't use padding or groups with Wavelet Conv")
                def _conv_class(**kwargs):
                    kwargs.pop("padding", None)
                    kwargs.pop("padding_mode", None)
                    kwargs.pop("groups", None)
                    return WTConv2d(**kwargs)
                conv_class = _conv_class

        if transposed:
            conv_class = nn.ConvTranspose2d

        self.conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
        )

        self.gating_conv = None
        if gating:
            self.gating_conv = conv_class(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                groups=groups,
            )
            #self.gating_mixer = ConvMixer(out_channels, activation=activation, batch_norm=batch_norm)
            #self.gating_mixer = ConvStrideMixer(out_channels, stride=16, activation=activation, hidden_channels=out_channels)
            self.gating_mixer = ConvDilationMixer(out_channels, activation=activation)
            self.gating_act = activation_to_module(gating_activation)

        if batch_norm and batch_norm_pos == 1:
            self.bn = nn.BatchNorm2d(out_channels)

        self.act = activation_to_module(activation)

        if batch_norm and batch_norm_pos == 2:
            self.bn = nn.BatchNorm2d(out_channels)

        if init_weights is not None:
            with torch.no_grad():
                for p in self.parameters():
                    if p.requires_grad:
                        p[:] = p * init_weights
                #self.conv.weight[:] = self.conv.weight * init_weights
                #if self.conv.bias is not None:
                #    self.conv.bias[:] = self.conv.bias * init_weights

    def forward(
            self,
            x: torch.Tensor,
            output_size: Union[None, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        original_x = x

        if self._batch_norm_pos == 0 and hasattr(self, "bn"):
            x = self.bn(x)

        if self.attention is not None:
            x = self.attention(x)

        gate_x = None
        if self.gating_conv is not None:
            gate_x = self.gating_conv(x)
            if output_size is not None and tuple(gate_x.shape[-2:]) != output_size:
                gate_x = F.pad(gate_x, (0, output_size[-1] - gate_x.shape[-1], 0, output_size[-2] - gate_x.shape[-2]))
            gate_x = self.gating_mixer(gate_x)
            if self.gating_act is not None:
                gate_x = self.gating_act(gate_x)

        x = self.conv(x)

        if output_size is not None and tuple(x.shape[-2:]) != output_size:
            x = F.pad(x, (0, output_size[-1] - x.shape[-1], 0, output_size[-2] - x.shape[-2]))

        if self._batch_norm_pos == 1 and hasattr(self, "bn"):
            x = self.bn(x)

        if self.act:
            x = self.act(x)

        if self._batch_norm_pos == 2 and hasattr(self, "bn"):
            x = self.bn(x)

        if gate_x is not None:
            x = x * gate_x

        if original_x.shape == x.shape:
            x = x + original_x

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
            conv_groups: Union[int, Iterable[int]] = 1,
            batch_norm: Union[bool, Iterable[bool]] = True,
            activation: Union[None, str, Callable] = "gelu",
            activation_last_layer: Union[None, str, Callable] = None,
            wavelet: bool = False,
            attention: Union[bool, Iterable[bool]] = False,
            gating: Union[bool, Iterable[bool]] = False,
            residual_weight: float = 1.,
            batch_norm_pos_encoder: int = 0,
            batch_norm_pos_decoder: int = 0,
            init_weights: Optional[float] = None,
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
        conv_groups = param_make_list(conv_groups, num_layers, "conv_groups")
        attention = param_make_list(attention, num_layers, "attention")
        gating = param_make_list(gating, num_layers, "gating")

        channels_list = [in_channels, *channels]

        self.encoder = nn.ModuleDict()

        with torch.no_grad():

            for i, (ch, ch_next, kernel_size, stride, pad, batch_norm, groups, attn, gate) in enumerate(
                    zip(channels_list, channels_list[1:], kernel_sizes, strides, paddings, batch_norms, conv_groups, attention, gating)
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
                    groups=groups,
                    init_weights=init_weights,
                    wavelet=wavelet,
                    attention=attn,
                    gating=gate,
                )

        channels_list = list(reversed([out_channels, *channels]))
        kernel_sizes = list(reversed(kernel_sizes))
        strides = list(reversed(strides))
        paddings = list(reversed(paddings))
        batch_norms = list(reversed(batch_norms))
        conv_groups = list(reversed(conv_groups))
        attention = list(reversed(attention))
        gating = list(reversed(gating))

        self.decoder = nn.ModuleDict()
        for i, (ch, ch_next, kernel_size, stride, pad, batch_norm, groups, attn, gate) in enumerate(
                zip(channels_list, channels_list[1:], kernel_sizes, strides, paddings, batch_norms, conv_groups, attention, gating)
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
                groups=groups,
                init_weights=init_weights,
                attention=attn,
                gating=gate,
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
