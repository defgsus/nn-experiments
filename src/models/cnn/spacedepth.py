"""
inspired by https://github.com/LabSAINT/SPD-Conv / https://arxiv.org/pdf/2208.03641.pdf
"""
from collections import OrderedDict

import torch
import torch.nn as nn


def space_to_depth(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"space_to_depth requires at least 3 dims, got {x.shape}")

    for dim in (-1, -2):
        s = x.shape[dim]
        if s / 2 != s // 2:
            raise ValueError(f"space_to_depth requires dimension {dim} to be divisible by 2, got {x.shape}")

    return torch.cat(
        [
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ],
        dim=-3,
    )


def depth_to_space(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"depth_to_space requires at least 3 dims, got {x.shape}")

    for dim in (-3,):
        s = x.shape[dim]
        if s / 4 != s // 4:
            raise ValueError(f"depth_to_space requires dimension {dim} to be divisible by 4, got {x.shape}")

    nc = x.shape[-3] // 2
    s = torch.stack([
        x[..., :nc, :, :],
        x[..., nc:, :, :],
    ], dim=-1).view(-1, nc, x.shape[-2], x.shape[-1] * 2)

    nc //= 2
    return torch.stack([
        s[..., :nc, :, :],
        s[..., nc:, :, :],
    ], dim=-2).view(-1, nc, x.shape[-2] * 2, x.shape[-1] * 2)


class SpaceToDepth(nn.Module):

    def __init__(self, transpose: bool = False):
        super().__init__()
        self.transpose = transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return depth_to_space(x)
        else:
            return space_to_depth(x)

    def extra_repr(self):
        return "transpose=True" if self.transpose else ""



class SpaceToDepthPool(nn.Module):
    """
    Replaces a Max/AvgPool2d with a stack of SpaceToDepth and Conv2d
    """
    def __init__(self, n_channels: int, kernel_size: int = 2):
        if kernel_size % 2 != 0:
            raise ValueError(f"`kernel_size` must be even, got {kernel_size}")

        super().__init__()

        layers = OrderedDict()
        for i in range(kernel_size // 2):
            layers[f"spd_{i + 1}"] = SpaceToDepth()
            layers[f"conv_{i + 1}"] = nn.Conv2d(n_channels * 4, n_channels, kernel_size=1)

        self.layers = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
