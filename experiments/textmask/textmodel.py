import math
from functools import partial
from typing import Union, Callable, Type, Dict, List, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_module, normalization_to_module
from src.models.attention import Attention1d
from src.util.params import param_make_tuple


class ConvTextLayer(nn.Module):
    def __init__(
            self,
            num_channels_in: int,
            num_channels_out: int,
            kernel_size: int,
            dilation: int,
            norm: Union[None, str, Type[nn.Module]],
            activation: Union[None, str, Callable],
            residual: bool,
            num_attention_heads: Union[bool, int] = 0,
            attention_invention: str = "QK",  # "QK", "QV", "KV", "QKV"
            attention_activation: Union[None, str, Callable] = "elu+1",
    ):
        super().__init__()
        self.residual = residual and num_channels_in == num_channels_out
        self.num_attention_heads = num_attention_heads
        self.attention_invention = attention_invention.upper()

        padding = int(math.floor(kernel_size / 2)) * dilation
        self.norm = normalization_to_module(norm, channels=num_channels_in)
        self.conv = nn.Conv1d(
            num_channels_in,
            num_channels_out * (len(attention_invention) if num_attention_heads else 1),
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.attn = None
        if num_attention_heads:
            if num_attention_heads is True:
                self.attn = Attention1d(
                    activation=attention_activation,
                )
            else:
                self.attn_mh = nn.MultiheadAttention(
                    embed_dim=num_channels_out,
                    num_heads=num_attention_heads,
                    batch_first=True,
                )
                self.attn = lambda q, k, v: self.attn_mh(q, k, v, need_weights=False)[0]

        self.act = activation_to_module(activation)

    def extra_repr(self) -> str:
        text = f"residual={self.residual}"
        if self.num_attention_heads:
            text = (
                f"{text}, num_attention_heads={self.num_attention_heads}"
                f", attention_invention='{self.attention_invention}'"
            )
        return text

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x

        if self.norm is not None:
            x = self.norm(x)

        y = self.conv(x)

        if self.attn is not None:
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

        if self.residual and y.shape == original_x.shape:
            y = y + original_x

        return y


class PositionEmbedding1d(nn.Module):
    def __init__(
            self,
            period: float = 20.,
    ):
        super().__init__()
        self.period = period

    def forward(self, length: int) -> torch.Tensor:
        phase = torch.arange(0, length) / self.period * math.pi * 2
        phase = phase * (1 + .02 * phase)
        return torch.stack([phase.sin(), phase.cos()])


class ConvTextModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_channels: int,
            kernel_size: Union[int, Iterable[int]] = 3,
            dilation: Union[int, Iterable[int]] = 1,
            norm: Union[None, str, Type[nn.Module]] = None,
            out_norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = None,
            num_attention_heads: Union[int, Iterable[int]] = 0,
            attention_activation: Union[None, str, Callable] = "elu+1",
            residual: Union[bool, Iterable[bool]] = True,
            residual_map: Dict[int, List[int]] = None,  # source layer -> target layer
            residual_map_concat: bool = False,
            pos_embedding: bool = False,
    ):
        super().__init__()

        self.residual = residual
        self.residual_map_concat = residual_map_concat

        self.layer_inputs: Dict[int, List[int]] = {}
        if residual_map:
            for l1, l2s in residual_map.items():
                for l2 in l2s:
                    assert l2 > l1 >= 0, f"Got l1={l1}, l2={l2}"
                    self.layer_inputs.setdefault(l2, []).append(l1)

        layer_output_channels = param_make_tuple(num_channels, num_layers, "num_channels")

        self.embedding = nn.Embedding(vocab_size, num_channels - (2 if pos_embedding else 0))
        self.pos_embedding = None
        if pos_embedding:
            self.pos_embedding = PositionEmbedding1d(period=10)
        self.layers = nn.ModuleList()

        ch = num_channels
        for i, next_ch, ks, dil, res, attn_heads in zip(
                range(num_layers),
                layer_output_channels,
                param_make_tuple(kernel_size, num_layers, "kernel_size"),
                param_make_tuple(dilation, num_layers, "dilation"),
                param_make_tuple(residual, num_layers, "residual"),
                param_make_tuple(num_attention_heads, num_layers, "num_attention_heads"),
        ):
            in_channels = ch
            if residual_map_concat:
                in_channels += sum(
                    layer_output_channels[l]
                    for l in self.layer_inputs.get(i, [])
                )
            self.layers.add_module(
                f"layer_{i+1}",
                ConvTextLayer(
                    num_channels_in=in_channels,
                    num_channels_out=next_ch,
                    kernel_size=ks,
                    dilation=dil,
                    norm=norm,
                    activation=activation,
                    residual=res,
                    num_attention_heads=attn_heads,
                    attention_activation=attention_activation,
                )
            )
            ch = next_ch

        self.out_norm = normalization_to_module(out_norm, channels=ch)
        self.lm_head = nn.Linear(ch, vocab_size, bias=False)
        if ch == num_channels:
            self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        x = x.permute(0, 2, 1)

        #print("X", x.shape, self.pos_embedding(x.shape[-1]).shape)
        if self.pos_embedding:
            x = torch.cat([
                x,
                self.pos_embedding(x.shape[-1])[None, :].to(x.device).expand(x.shape[0], -1, -1),
            ], dim=1)
            #print("XXX", x.shape)

        layer_out_map = {}
        for idx, layer in enumerate(self.layers):
            inp = x
            if self.layer_inputs.get(idx):
                if self.residual_map_concat:
                    inp = torch.cat([
                        inp,
                        *(layer_out_map[i] for i in self.layer_inputs[idx]),
                    ], dim=-2)
                else:
                    for i in self.layer_inputs[idx]:
                        inp = inp + layer_out_map[i]

            layer_out_map[idx] = x = layer(inp)

        x = x.permute(0, 2, 1)

        if self.out_norm is not None:
            x = self.out_norm(x)

        if self.pos_embedding is not None:
            x = x[..., :-2]

        logits = self.lm_head(x)

        return logits
