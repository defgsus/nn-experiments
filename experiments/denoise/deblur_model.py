import math
import dataclasses
from typing import Tuple

import torch
import torch.nn as nn

from src.models.util import *


class ResConvBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            channels_out: Optional[int] = None,
            kernel_size: int = 3,
            act: str = "gelu",
    ):
        super().__init__()

        if channels_out is None:
            channels_out = channels

        self.conv = nn.Conv2d(
            channels, channels_out, kernel_size, padding=int(math.floor(kernel_size / 2))
        )
        self.act = activation_to_module(act)

    def forward(self, x):
        y = self.conv(x)
        if self.act is not None:
            y = self.act(y)

        if y.shape[-3] == x.shape[-3]:
            y = y + x

        return y


class UNet(nn.Module):


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



class DeblurModel(nn.Module):

    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            channels_hidden: int,
            num_layers: int = 1,
            kernel_size: int = 3,
            act: str = "gelu",
            attention: Union[bool, int] = False,
            attention_heads: int = 0,
            attention_dropout: float = 0.,
            attention_residual: bool = True,
            norm: Optional[str] = None,
            depth_conv: bool = False,
    ):
        super().__init__()

        padding = int(math.floor(kernel_size / 2))
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_hidden, kernel_size=kernel_size, padding=padding),
        )
        if act is not None:
            self.layers.append(activation_to_module(act))

        for i in range(num_layers):
            if isinstance(attention, bool):
                do_attention = attention
            else:
                do_attention = attention > 0 and i % attention == 0

            if do_attention:
                if attention_heads > 0:
                    self.layers.append(SelfAttention(
                        channels=channels_hidden,
                        heads=attention_heads,
                        dropout=attention_dropout,
                        residual=attention_residual,
                    ))
                else:
                    self.layers.append(AttentionConvBlock(
                        channels_hidden, channels_hidden, kernel_size=kernel_size, activation=act, dropout=attention_dropout,
                        residual=attention_residual,
                        norm=norm, # depth_conv=depth_conv,
                    ))
                #else:
                #    self.layers.append(ResidualAdd(
                #        LiteMLA(channels_hidden, channels_hidden)
                #    ))

            else:
                self.layers.append(ConvBlock(channels_hidden, kernel_size=kernel_size, act=act, depth_conv=depth_conv))

        self.layers.append(
            nn.Conv2d(channels_hidden, channels_out, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
