import math
import dataclasses
from typing import Tuple

import torch
import torch.nn as nn

from src.models.util import *
from src.models.efficientvit.ops import LiteMLA


class Conv2dDepth(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *args, **kwargs,
    ):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.point_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(x)
        y = self.point_conv(y)

        return y


def _get_conv_class(depth_conv: bool):
    return Conv2dDepth if depth_conv else nn.Conv2d


class ConvBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            act: str = "gelu",
            depth_conv: bool = False,
    ):
        super().__init__()
        self.conv = _get_conv_class(depth_conv)(
            channels, channels, kernel_size, padding=int(math.floor(kernel_size / 2))
        )
        self.act = activation_to_module(act)

    def forward(self, x):
        y = self.conv(x)
        if self.act is not None:
            y = self.act(y)
        return y + x


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


class AttentionConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: Optional[int] = None,
            activation: Union[None, str, Callable] = None,
            dropout: float = 0.,
            residual: bool = True,
            norm: Optional[str] = None,
            depth_conv: bool = False,
    ):
        super().__init__()
        self._act = activation
        self._residual = residual

        if padding is None:
            padding = int(math.floor(kernel_size / 2))

        self.q = _get_conv_class(depth_conv)(in_channels, out_channels, kernel_size, padding=padding)
        self.k = _get_conv_class(depth_conv)(in_channels, out_channels, kernel_size, padding=padding)
        self.v = _get_conv_class(depth_conv)(in_channels, out_channels, kernel_size, padding=padding)
        self.act = activation_to_callable(activation)
        self.norm = normalization_to_module(norm, out_channels)

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)

    def extra_repr(self):
        return f"residual={self._residual}, activation={repr(self._act)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        if self.norm is not None:
            x = self.norm(x)

        q = F.relu(self.q(x))
        k = F.relu(self.k(x))
        v = self.v(x)

        attn = q @ k
        attn = F.softmax(attn.view(B, -1, H * W), dim=-1).view(B, -1, H, W)
        # attn = torch.max(F.softmax(attn, dim=-1), F.softmax(attn, dim=-2))
        if hasattr(self, "dropout"):
            attn = self.dropout(attn)

        y = attn @ v

        if self.act is not None:
            y = self.act(y)

        if self._residual:
            y = y + residual

        return y


class AttentionConvBlockMultiScale(nn.Module):
    """
    Experiments from EfficientViT (https://arxiv.org/abs/2205.14756)
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: Optional[int] = None,
            activation: Union[None, str, Callable] = None,
            dropout: float = 0.,
            residual: bool = False,
            norm: Optional[str] = None,
    ):
        super().__init__()
        self._act = activation
        self._residual = residual

        if padding is None:
            padding = int(math.floor(kernel_size / 2))

        self.q = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.k = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.v = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.attn_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.attn_conv5 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.act = activation_to_callable(activation)
        self.norm = normalization_to_module(norm, out_channels)

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)

    def extra_repr(self):
        return f"residual={self._residual}, activation={repr(self._act)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        if self.norm is not None:
            x = self.norm(x)

        q = F.relu(self.q(x))
        k = F.relu(self.k(x))
        v = self.v(x)

        attn = q @ k
        if hasattr(self, "dropout"):
            attn = self.dropout(attn)

        attn = attn + self.attn_conv3(attn) + self.attn_conv5(attn)

        y = attn @ v

        if self.act is not None:
            y = self.act(y)

        if self._residual:
            y = y + residual

        return y


class ResConvLayers(nn.Module):

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
