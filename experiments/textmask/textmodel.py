import math
from typing import Union, Callable, Type, Dict, List, Iterable

import torch
import torch.nn as nn

from src.models.util import activation_to_module, normalization_to_module
from src.util.params import param_make_tuple


class ConvTextLayer(nn.Module):
    def __init__(
            self,
            num_channels_in: int,
            num_channels_out: int,
            kernel_size: int,
            padding: int,
            norm: Union[None, str, Type[nn.Module]],
            activation: Union[None, str, Callable],
            residual: bool,
    ):
        super().__init__()
        self.residual = residual and num_channels_in == num_channels_out

        self.norm = normalization_to_module(norm, channels=num_channels_in)
        self.conv = nn.Conv1d(num_channels_in, num_channels_out, kernel_size=kernel_size, padding=padding)
        self.act = activation_to_module(activation)

    def extra_repr(self) -> str:
        return f"residual={self.residual}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x

        if self.norm is not None:
            x = self.norm(x)

        y = self.conv(x)

        if self.act is not None:
            y = self.act(y)

        if self.residual and y.shape == x.shape:
            y = y + original_x

        return y


class ConvTextModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_channels: Union[int, Iterable[int]],
            kernel_size: Union[int, Iterable[int]] = 3,
            norm: Union[None, str, Type[nn.Module]] = None,
            out_norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = None,
            residual: Union[bool, Iterable[bool]] = True,
            residual_map: Dict[int, List[int]] = None,  # source layer -> target layer
    ):
        super().__init__()

        self.residual = residual

        self.layer_inputs: Dict[int, List[int]] = {}
        if residual_map:
            for l1, l2s in residual_map.items():
                for l2 in l2s:
                    assert l2 > l1 >= 0, f"Got l1={l1}, l2={l2}"
                    self.layer_inputs.setdefault(l2, []).append(l1)

        layer_output_channels = param_make_tuple(num_channels, num_layers, "num_channels")

        self.embedding = nn.Embedding(vocab_size, num_channels)
        self.layers = nn.ModuleList()

        ch = num_channels
        for i, next_ch, ks, res in zip(
                range(num_layers),
                layer_output_channels,
                param_make_tuple(kernel_size, num_layers, "kernel_size"),
                param_make_tuple(residual, num_layers, "residual"),
        ):
            self.layers.add_module(
                f"layer_{i+1}",
                ConvTextLayer(
                    num_channels_in=ch + sum(
                        layer_output_channels[l]
                        for l in self.layer_inputs.get(i, [])
                    ),
                    num_channels_out=next_ch,
                    kernel_size=ks,
                    padding=int(math.floor(ks / 2)),
                    norm=norm,
                    activation=activation,
                    residual=res,
                )
            )
            ch = next_ch

        self.out_norm = normalization_to_module(out_norm, channels=num_channels)
        self.lm_head = nn.Linear(num_channels, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        x = x.permute(0, 2, 1)

        layer_out_map = {}
        for idx, layer in enumerate(self.layers):
            inp = x
            if self.layer_inputs.get(idx):
                inp = torch.cat([
                    inp,
                    *(layer_out_map[i] for i in self.layer_inputs[idx]),
                ], dim=-2)

            layer_out_map[idx] = x = layer(inp)

        x = x.permute(0, 2, 1)

        if self.out_norm is not None:
            x = self.out_norm(x)

        logits = self.lm_head(x)

        return logits
