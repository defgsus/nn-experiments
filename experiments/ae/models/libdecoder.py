import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_callable


class LibDecoder2d(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            lib_size: int,
            patch_size: Union[int, Tuple[int, int]],
            patch_filename: Optional[str] = None,
            output_activation: Union[None, str, Callable] = "sigmoid",
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = code_size
        self.lib_size = lib_size
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]

        self.grid_shape = (shape[-2] // patch_size[-2], shape[-1] // patch_size[-1])

        self.mlp = nn.Sequential(
            nn.Linear(code_size, lib_size * math.prod(self.grid_shape)),
            #nn.Linear(code_size, lib_size * math.prod(self.grid_shape) // 2),
            #nn.LeakyReLU(inplace=True),
            #nn.Linear(lib_size * math.prod(self.grid_shape) // 2, lib_size * math.prod(self.grid_shape)),
            #nn.Sigmoid(), # NOT GOOD after single layer
        )

        self.conv = nn.ConvTranspose2d(lib_size, shape[0], kernel_size=patch_size, stride=patch_size)
        if patch_filename:
            with torch.no_grad():
                patches = torch.load(patch_filename)
                self.conv.weight[:patches.shape[0]] = patches

        self.output_activation = activation_to_callable(output_activation)

    def forward(self, x):
        grid = self.mlp(x).reshape(-1, self.lib_size, *self.grid_shape)
        return self.output_activation(self.conv(grid))

    def extra_repr(self):
        return f"shape={self.shape}, code_size={self.code_size}, lib_size={self.lib_size}"

