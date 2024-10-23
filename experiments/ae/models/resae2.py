import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module


class ResampleBlock(nn.Module):

    def __init__(
            self,
            num_channels: int,
            kernel_size: int = 3,
            down: bool = True,
            batch_norm: bool = True,
            activation: Union[None, str, Callable] = None,
    ):
        assert kernel_size % 2 == 1, f"Must have odd `kernel_size`, got {kernel_size}"
        super().__init__()
        self._down = down
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_channels * 2 if down is False else num_channels)
        if self._down:
            self.conv = nn.Conv2d(num_channels, num_channels * 2, kernel_size, padding=int(math.floor(kernel_size / 2)), stride=2)
        else:
            self.conv = nn.ConvTranspose2d(num_channels, num_channels * 2, kernel_size, padding=int(math.floor(kernel_size / 2)), stride=2)
        self.act = activation_to_module(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.shape
        B, C, H, W = x.shape
        assert C / 2 == C // 2, x.shape

        if self.bn is not None:
            x = self.bn(x)

        if self._down:
            r = F.pixel_unshuffle(x, 2)
            r = (r[:, :C*2] + r[:, C*2:]) / 2.
        else:
            r = F.pixel_shuffle(x, 2)
            r = torch.concat([r, r], dim=-3)
            return r

        out = self.conv(x) + r
        if self.act is not None:
            out = self.act(out)

        return out


class ResAE(nn.Module):

    def __init__(
            self,
            input_channels: int,
            num_channels: int,
            num_encoded_channels: Optional[int] = None,
            kernel_size: int = 3,
            num_layers: int = 4,
            batch_norm: bool = True,
            activation: Union[None, str, Callable] = "gelu",
            verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.input = nn.Conv2d(input_channels, num_channels, 1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        ch = num_channels
        for i in range(num_layers):
            dub_chan = True #i % 2 == 1
            self.down_blocks.append(ResampleBlock(
                ch, kernel_size, down=True, batch_norm=batch_norm,
                activation=activation if i != num_layers - 1 else None,
            ))
            if not dub_chan:
                self.down_blocks.append(nn.Conv2d(ch * 2, ch, 1))
            self.up_blocks.insert(0, ResampleBlock(
                ch, kernel_size, down=False, batch_norm=batch_norm,
                activation=activation if i != 0 else None
            ))
            if not dub_chan:
                self.up_blocks.insert(0, nn.ConvTranspose2d(ch, ch * 2, 1))
            if dub_chan:
                ch *= 2
        self.output = nn.Conv2d(num_channels, input_channels, 1)

        self.encoder_conv = None
        self.decoder_conv = None
        if num_encoded_channels is not None and ch != num_encoded_channels:
            self.encoder_conv = nn.Conv2d(ch, num_encoded_channels, 1)
            self.decoder_conv = nn.ConvTranspose2d(num_encoded_channels, ch, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input(x)
        for b in self.down_blocks:
            if self.verbose:
                bs = str(b).replace('\n', ' ')
                print(f"{y.shape} -> {bs}")
            y = b(y)
        if self.verbose:
            print(y.shape)
        if self.encoder_conv is not None:
            y = self.encoder_conv(y)
        return y

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.decoder_conv is not None:
            y = self.decoder_conv(y)
        for b in self.up_blocks:
            if self.verbose:
                bs = str(b).replace('\n', ' ')
                print(f"{y.shape} -> {bs}")
            y = b(y)
        if self.verbose:
            print(y.shape)
        return self.output(y)

    def forward(self, x):
        return self.decode(self.encode(x))
