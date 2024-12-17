import math
from typing import Union, Callable, Type, Dict, List, Iterable

import torch
import torch.nn as nn

from src.models.util import activation_to_module, normalization_to_module
from src.util.params import param_make_tuple


class ConvTextLayer(nn.Module):
    def __init__(
            self,
            num_layers_in: int,
            num_channels_in: int,
            num_channels_out: int,
            kernel_size: int,
            dilation: int,
            norm: Union[None, str, Type[nn.Module]],
            activation: Union[None, str, Callable],
            dropout: float = 0.,
    ):
        super().__init__()

        self.dropout = None
        self.input_weight = None
        in_channels = num_channels_in

        if num_layers_in > 1:
            in_channels *= 2
            self.input_weight = nn.Parameter(
                torch.randn(2, num_layers_in, num_channels_in, 1) / 100.
            )

        self.norm = normalization_to_module(norm, channels=in_channels)

        if dropout:
            self.dropout = nn.Dropout1d(dropout)
        self.conv = nn.Conv1d(
            in_channels,
            num_channels_out,
            kernel_size=kernel_size,
            padding=((kernel_size - 1) // 2) * dilation,
            dilation=dilation,
        )
        self.act = activation_to_module(activation)

    def extra_repr(self) -> str:
        return f"num_layers_in={1 if self.input_weight is None else self.input_weight.shape[0]}"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:

        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = None
            for i, input in enumerate(inputs):
                input = torch.cat([
                    self.input_weight[0, i] * input,
                    self.input_weight[1, i] * input,
                ], dim=-2)
                if x is None:
                    x = input
                else:
                    x = x + input

        if self.norm is not None:
            x = self.norm(x)

        if self.dropout is not None:
            x = self.dropout(x)

        y = self.conv(x)
        # print(f"X -> Y: {x.shape} {y.shape}")

        if self.act is not None:
            y = self.act(y)

        return y


class ConvTextModel2(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_channels: Union[int, Iterable[int]],
            kernel_size: Union[int, Iterable[int]] = 3,
            dilation: Union[int, Iterable[int]] = 1,
            norm: Union[None, str, Type[nn.Module]] = None,
            out_norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = None,
            dropout: float = 0.,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, num_channels)
        self.layers = nn.ModuleList()

        for i, ks, dil in zip(
                range(num_layers),
                param_make_tuple(kernel_size, num_layers, "kernel_size"),
                param_make_tuple(dilation, num_layers, "dilation"),
        ):
            self.layers.add_module(
                f"layer_{i+1}",
                ConvTextLayer(
                    # 1st layer gets input, 2nd layer gets input and 1st layer output...
                    num_layers_in=i + 1,
                    num_channels_in=num_channels,
                    num_channels_out=num_channels,
                    kernel_size=ks,
                    dilation=dil,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )

        self.out_norm = normalization_to_module(out_norm, channels=num_channels)
        self.lm_head = nn.Linear(num_channels, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        x = x.permute(0, 2, 1)

        inputs = [x]
        for idx, layer in enumerate(self.layers):
            x = layer(*inputs)
            inputs.append(x)

        x = x.permute(0, 2, 1)

        if self.out_norm is not None:
            x = self.out_norm(x)

        logits = self.lm_head(x)

        return logits
