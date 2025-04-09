import math
from functools import partial
from typing import Union, Callable, Type, Dict, List, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_module, normalization_to_module
from src.models.attention import Attention1d
from src.models.cnn import CheapConv1d
from src.models.encoder import DiagonalEmbedding
from src.util.params import param_make_tuple


class AttentionLayer(nn.Module):
    def __init__(
            self,
            num_channels_in: int,
            num_channels_out: int,
            activation: Union[None, str, Callable] = "gelu",
            attention_invention: str = "QK",  # "QK", "QV", "KV", "QKV"
            attention_activation: Union[None, str, Callable] = "elu+1",
            cheap: bool = False,
    ):
        super().__init__()
        self.attention_invention = attention_invention.upper()
        self.attention_activation = attention_activation

        self.conv = (CheapConv1d if cheap else nn.Conv1d)(
            num_channels_in,
            num_channels_out * len(attention_invention),
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        self.attn = Attention1d(
            activation=attention_activation,
        )
        self.act = activation_to_module(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x

        y = self.conv(x)

        # channel-wise attention (put channels into last dim)
        y = y.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        if self.attention_invention in ("QK", "KQ"):
            q, k = torch.split(y, y.shape[-1] // 2, dim=-1)
            y = self.attn(q, k, x)
        elif self.attention_invention in ("QV", "VQ"):
            q, v = torch.split(y, y.shape[-1] // 2, dim=-1)
            y = self.attn(q, x, v)
        elif self.attention_invention in ("KV", "VK"):
            k, v = torch.split(y, y.shape[-1] // 2, dim=-1)
            y = self.attn(x, k, v)
        elif self.attention_invention in ("QKV", "QVK", "KQV", "KVQ", "VQK", "VKQ"):
            q, k, v = torch.split(y, y.shape[-1] // 3, dim=-1)
            y = self.attn(q, k, v)
        else:
            raise AssertionError(f"Invalid `attention_invention` '{self.attention_invention}'")

        y = y.permute(0, 2, 1)

        if self.act is not None:
            y = self.act(y)

        return y + original_x


class ConvTextLayer(nn.Module):
    def __init__(
            self,
            num_channels_in: int,
            num_channels_out: int,
            kernel_size: int,
            dilation: int,
            dim_reduction: int,
            norm: Union[None, str, Type[nn.Module]],
            activation: Union[None, str, Callable],
            residual: bool,
            permute: bool,
            cheap: bool = False,
    ):
        super().__init__()
        self.residual = residual and num_channels_in == num_channels_out
        self.permute = permute

        padding = int(math.floor(kernel_size / 2)) * dilation - dim_reduction
        self.norm = normalization_to_module(norm, channels=num_channels_in)
        self.conv = (CheapConv1d if cheap else nn.Conv1d)(
            num_channels_in,
            num_channels_out,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.act = activation_to_module(activation)

    def extra_repr(self) -> str:
        text = f"residual={self.residual}, permute={self.permute}"
        return text

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x

        if self.norm is not None:
            x = self.norm(x)

        if self.permute:
            x = x.permute(0, 2, 1)

        y = self.conv(x)

        if self.permute:
            y = y.permute(0, 2, 1)

        if self.act is not None:
            y = self.act(y)

        if self.residual and y.shape[:2] == original_x.shape[:2]:
            y = y + original_x[:, :, :y.shape[-1]]

        return y


class ConvTextModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            num_layers: int,
            num_channels: Union[int, Iterable[int]],
            kernel_size: Union[int, Iterable[int]] = 3,
            dilation: Union[int, Iterable[int]] = 1,
            dim_reduction: int = 0,
            norm: Union[None, str, Type[nn.Module]] = None,
            out_norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = None,
            cheap: Union[bool, Iterable[bool]] = False,
            permute: Union[None, str, List[str]] = None,
            residual: Union[bool, Iterable[bool]] = True,
            attention_at: Tuple[int, ...] = tuple(),
            attention_invention: str = "QK",  # "QK", "QV", "KV", "QKV"
            head_type: str = "full",  # "full", "last<N>"
    ):
        super().__init__()

        self.residual = residual
        self.head_type = head_type
        num_channels = param_make_tuple(num_channels, num_layers, "num_channels")

        self.embedding = nn.Embedding(
            vocab_size,
            num_channels[0],
        )

        self.layers = nn.Sequential()

        ch = num_channels[0]
        do_permute = permute == "odd"
        reduction_per_permute = {
            True: 0,
            False: 0,
        }
        for i, next_ch, ks, dil, res, cheap_ in zip(
                range(num_layers),
                num_channels,
                param_make_tuple(kernel_size, num_layers, "kernel_size"),
                param_make_tuple(dilation, num_layers, "dilation"),
                param_make_tuple(residual, num_layers, "residual"),
                param_make_tuple(cheap, num_layers, "cheap"),
        ):
            is_last_layer = i == num_layers - 1
            in_channels = ch
            self.layers.add_module(
                f"layer_{i+1}",
                ConvTextLayer(
                    num_channels_in=(seq_length if do_permute else in_channels) - reduction_per_permute[do_permute],
                    num_channels_out=(seq_length if do_permute and not is_last_layer else next_ch) - reduction_per_permute[do_permute],
                    kernel_size=ks,
                    dilation=dil,
                    dim_reduction=dim_reduction,
                    norm=None if is_last_layer else norm,
                    activation=None if is_last_layer else activation,
                    residual=res,
                    cheap=cheap_,
                    permute=do_permute,
                )
            )
            if dim_reduction:
                reduction_per_permute[not do_permute] += dim_reduction * 2

            if i in attention_at:
                self.layers.add_module(
                    f"attn_{i+1}",
                    AttentionLayer(
                        num_channels_in=seq_length if permute else next_ch - reduction_per_permute[do_permute],
                        num_channels_out=seq_length if permute else next_ch - reduction_per_permute[do_permute],
                        attention_invention=attention_invention,
                        #attention_activation=
                        cheap=cheap_,
                    )
                )
            if permute in ("odd", "even") and not is_last_layer:
                do_permute = not do_permute
            ch = next_ch

        self.out_norm = normalization_to_module(out_norm, channels=ch)

        if self.head_type == "full":
            with torch.no_grad():
                out_shape = self.layers(torch.zeros(2, num_channels[0], seq_length)).shape[1:]

            self.head = nn.Linear(math.prod(out_shape), vocab_size)

        elif self.head_type.startswith("last"):
            n = int(self.head_type[4:])
            self.head = nn.Linear((ch - reduction_per_permute[do_permute]) * n, vocab_size)

        else:
            raise ValueError(f"Unknown head_type '{self.head_type}'")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).permute(0, 2, 1)

        x = self.layers(x)

        if self.out_norm is not None:
            x = self.out_norm(x)

        if self.head_type == "full":
            logits = self.head(x.flatten(1))

        elif self.head_type.startswith("last"):
            n = int(self.head_type[4:])
            logits = self.head(x[:, :, -n:].flatten(1))

        return logits
