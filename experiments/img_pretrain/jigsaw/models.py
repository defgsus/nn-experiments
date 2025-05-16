import math
from typing import Tuple

import torch
import torch.nn as nn

import datasets
datasets.load_dataset()

class ConvStage(nn.Module):

    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            kernel_size: int = 7,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in, channels_out, kernel_size,
            #padding=(kernel_size + 1) // 2,
        )
        #self.pool = nn.MaxPool2d(kernel_size=2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        orig_x = x
        x = self.conv(x)
        #x = self.pool(x)
        x = self.act(x)
        if x.shape == orig_x.shape:
            x = x + orig_x
        return x


class JigsawModel(nn.Module):

    def __init__(
            self,
            patch_shape: Tuple[int, int, int],
            num_patches: int,
            num_classes: int,
            channels: Tuple[int],
            kernel_size: int = 7,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.encoder = nn.Sequential()
        for i, (ch, next_ch) in enumerate(zip(channels, channels[1:])):
            self.encoder.add_module(f"conv_{i+1}", ConvStage(ch, next_ch, kernel_size=kernel_size))

        with torch.no_grad():
            out_shape = self.encoder(torch.zeros(1, *patch_shape)).shape
            self.feature_size = math.prod(out_shape)
            out_size = self.num_patches * self.feature_size

        self.head = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.GELU(),
            nn.Linear(out_size // 2, self.num_classes),
        )

    def forward(self, patches: torch.Tensor):
        B, P, C, H, W = patches.shape
        assert P == self.num_patches, f"Expected {self.num_patches}, got {P}"

        patches = patches
        features = self.encoder(patches.reshape(-1, C, H, W)).reshape(B, -1)
        return self.head(features)


