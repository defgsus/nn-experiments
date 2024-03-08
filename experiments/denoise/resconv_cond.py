import unittest
import random
import math
import time
from io import BytesIO
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.util import *
from src.models.util import *


class ConditionalConvLayer(nn.Module):

    def extra_repr(self):
        return (
            f"in_channels={self._in_channels}, out_channels={self._out_channels}"
            f", condition_size={self._condition_size}, padding_mode='{self.conv.padding_mode}'"
        )

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            condition_size: Optional[int] = None,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            batch_norm: bool = True,
            batch_norm_pos: int = 0,
            activation: Union[None, str, Callable] = "gelu",
            padding_mode: str = "zeros",
            transposed: bool = False,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._condition_size = condition_size
        self._batch_norm_pos = batch_norm_pos

        if batch_norm and batch_norm_pos == 0:
            self.bn = nn.BatchNorm2d(in_channels)

        self.conv = (nn.ConvTranspose2d if transposed else nn.Conv2d)(
            in_channels=self._in_channels + (self._condition_size or 0),
            out_channels=self._out_channels,
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

    def forward(
            self,
            x: torch.Tensor,
            condition: Optional[torch.Tensor] = None,
            *,
            output_size: Union[None, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self._batch_norm_pos == 0 and hasattr(self, "bn"):
            x = self.bn(x)

        if self._condition_size:
            B, C, H, W = x.shape
            if condition is None:
                condition_map = torch.zeros(B, self._condition_size, H, W)
            else:
                # B, C -> B, C, H, W
                condition_map = condition[:, :, None, None].expand(-1, -1, H, W)

            x = torch.concat([x, condition_map.to(x)], dim=-3)

        x = self.conv(x)

        if output_size is not None and tuple(x.shape[-2:]) != output_size:
            x = F.pad(x, (0, output_size[-1] - x.shape[-1], 0, output_size[-2] - x.shape[-2]))

        if self._batch_norm_pos == 1 and hasattr(self, "bn"):
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        if self._batch_norm_pos == 2 and hasattr(self, "bn"):
            x = self.bn(x)

        return x


class ConditionalResConv(nn.Module):

    def extra_repr(self):
        return (
            f"in_channels={self._in_channels}, out_channels={self._out_channels}, num_layers={self._num_layers}"
            f", condition_size={self._condition_size}, padding_mode='{self._padding_mode}'"
        )

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            channels: Union[int, Iterable[int]] = 32,
            condition_size: Optional[int] = None,
            internal_condition_size: Optional[Union[int, Iterable[Optional[int]]]] = None,
            condition_last_decoder_layer: bool = False,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: Union[int, Iterable[int]] = 1,
            padding: Union[int, Iterable[int]] = 0,
            padding_mode: str = "zeros",
            conv_groups: Union[int, Iterable[int]] = 1,
            batch_norm: Union[bool, Iterable[bool]] = True,
            activation: Union[None, str, Callable] = "gelu",
            activation_last_layer: Union[None, str, Callable] = None,
            residual_weight: float = 1.,
            batch_norm_pos_encoder: int = 0,
            batch_norm_pos_decoder: int = 0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_layers = num_layers
        self._residual_weight = residual_weight
        self._condition_size = condition_size
        self._condition_last_decoder_layer = condition_last_decoder_layer
        self._padding_mode = padding_mode

        channels = param_make_list(channels, num_layers, "channels")
        kernel_sizes = param_make_list(kernel_size, num_layers, "kernel_size")
        strides = param_make_list(stride, num_layers, "stride")
        paddings = param_make_list(padding, num_layers, "padding")
        batch_norms = param_make_list(batch_norm, num_layers, "batch_norm")
        conv_groups = param_make_list(conv_groups, num_layers, "conv_groups")

        # --- conditional projections ---

        self.condition_projections = None

        if condition_size is None:
            if internal_condition_size is not None:
                raise ValueError(
                    f"`condition_size` is not set but `internal_condition_size` is set to {internal_condition_size}"
                )
            self._internal_condition_sizes = [None] * num_layers
        else:
            if internal_condition_size is None:
                internal_condition_size = condition_size
            self._internal_condition_sizes = param_make_list(
                internal_condition_size, num_layers, "internal_condition_size"
            )

            if any(ics != condition_size for ics in self._internal_condition_sizes):
                self.condition_projections = nn.ModuleDict()
                for ics in self._internal_condition_sizes:
                    if ics is not None and ics != condition_size:
                        key = str(ics)
                        if key not in self.condition_projections:
                            self.condition_projections[key] = nn.Linear(condition_size, ics)

        # --- encoder ---

        self.encoder = nn.ModuleDict()

        channels_list = [in_channels, *channels]

        for i, (ch, ch_next, kernel_size, stride, pad, batch_norm, groups, condition_size) in enumerate(zip(
                channels_list, channels_list[1:],
                kernel_sizes, strides, paddings, batch_norms, conv_groups, self._internal_condition_sizes
        )):
            self.encoder[f"layer_{i + 1}"] = ConditionalConvLayer(
                in_channels=ch,
                out_channels=ch_next,
                kernel_size=kernel_size,
                condition_size=condition_size,
                stride=stride,
                padding=pad,
                batch_norm=batch_norm and i < num_layers - 1,
                activation=(activation if i < num_layers - 1 else None),
                padding_mode=padding_mode,
                batch_norm_pos=batch_norm_pos_encoder,
                groups=groups,
            )

        # --- decoder ---

        channels_list = list(reversed([out_channels, *channels]))
        kernel_sizes = list(reversed(kernel_sizes))
        strides = list(reversed(strides))
        paddings = list(reversed(paddings))
        batch_norms = list(reversed(batch_norms))
        conv_groups = list(reversed(conv_groups))
        internal_condition_sizes = list(reversed(self._internal_condition_sizes))

        self.decoder = nn.ModuleDict()
        for i, (ch, ch_next, kernel_size, stride, pad, batch_norm, groups, condition_size) in enumerate(zip(
                channels_list, channels_list[1:],
                kernel_sizes, strides, paddings, batch_norms, conv_groups, internal_condition_sizes,
        )):
            self.decoder[f"layer_{i + 1}"] = ConditionalConvLayer(
                in_channels=ch,
                out_channels=ch_next,
                kernel_size=kernel_size,
                condition_size=condition_size,
                stride=stride,
                padding=pad,
                batch_norm=batch_norm and i < num_layers - 1,
                activation=activation if i < num_layers - 1 else activation_last_layer,
                padding_mode=padding_mode,
                transposed=True,
                batch_norm_pos=batch_norm_pos_decoder,
                groups=groups,
            )

    def forward(
            self,
            x: torch.Tensor,
            condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected `x` to have shape (B, C, H, W), got {x.shape}")
        B, C, H, W = x.shape

        condition_per_layer = self._get_condition_per_layer(B, condition)

        # --- encoder ---

        state = x
        state_history = [state]
        for idx, (layer, cond) in enumerate(zip(self.encoder.values(), condition_per_layer)):
            state = layer(state, condition=cond)
            state_history.append(state)

        # --- decoder ---

        for idx, (layer, cond) in enumerate(zip(self.decoder.values(), reversed(condition_per_layer))):
            if idx > 0:
                state = state + self._residual_weight * state_history[-(idx + 1)]

            if not self._condition_last_decoder_layer and idx == len(self.decoder) - 1:
                cond = None

            state = layer(state, condition=cond, output_size=state_history[-(idx + 2)].shape[-2:])

        return state

    def _get_condition_per_layer(
            self,
            batch_size: int,
            condition: Optional[torch.Tensor] = None,
    ) -> List[Optional[torch.Tensor]]:
        num_layers = len(self.encoder)

        if condition is None:
            return [None] * num_layers

        if self._condition_size is None:
            raise ValueError(f"Passing `condition` requires initialization with `condition_size` > 0")

        if condition.ndim == 1:
            condition = condition[None, :].expand(batch_size, condition.shape[0])

        if condition.shape != torch.Size((batch_size, self._condition_size)):
            raise ValueError(
                f"Expected `condition` of shape (B={batch_size}, {self._condition_size}), got {condition.shape}"
            )

        if self.condition_projections is None:
            return [condition] * num_layers

        projected_conditions = {
            None: None,
            self._condition_size: condition
        }
        for ics, proj in self.condition_projections.items():
            projected_conditions[int(ics)] = proj(condition)

        return [
            projected_conditions[ics]
            for ics in self._internal_condition_sizes
        ]


class TestConditionalResConv(unittest.TestCase):

    @torch.no_grad()
    def test_resconv(self):
        for params in tqdm(list(iter_parameter_permutations({
            "kernel_size": [1, 2, 3, [1, 2, 3]],
            "stride": [1, 2, 3, [1, 2, 3]],
            "padding": [0, 1, 2, [0, 1, 2]],
            "channels": [32, [16, 32, 48]],
            "shape": [
                (3, 32, 32),
                (1, 31, 33),
            ],
        }))):
            msg = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
            shape = params["shape"]

            model = ConditionalResConv(
                in_channels=shape[0],
                channels=params["channels"],
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

    @torch.no_grad()
    def test_resconv_conditional(self):
        for params in tqdm(list(iter_parameter_permutations({
            "kernel_size": [3],
            "stride": [1],
            "padding": [0],
            "channels": [32, [16, 32, 48], [16, 32, 48, 16]],
            "shape": [
                (3, 32, 32),
                (1, 31, 33),
            ],
            "condition_size": [1, 10, 128],
            "internal_condition_size": [None, 10, 16, [1, 10, 16], [128, None, 24, 12]],
        }))):
            len_channels = len(params["channels"]) if isinstance(params["channels"], list) else None
            len_cond_size = len(params["internal_condition_size"]) if isinstance(params["internal_condition_size"], list) else None
            if len_channels != len_cond_size:
                continue

            msg = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
            shape = params["shape"]

            model = ConditionalResConv(
                in_channels=shape[0],
                num_layers=len(params["channels"]) if isinstance(params["channels"], list) else 3,
                channels=params["channels"],
                condition_size=params["condition_size"],
                internal_condition_size=params["internal_condition_size"],
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
                activation_last_layer="sigmoid",
            ).eval()
            for batch_size in (1, 3):
                input = torch.rand(batch_size, *shape)
                condition = torch.rand(batch_size, params["condition_size"])

                output = model(input, condition)

            self.assertEqual(
                input.shape,
                output.shape,
                msg,
            )

        print(model)
