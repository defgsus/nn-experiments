import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_callable, activation_to_module
from src.models.transform import Reshape


class ResMLP(nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_hidden: int,
            activation: Union[None, str, Callable] = "relu",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_hidden = num_hidden

        self.mlp = nn.Sequential()
        self.mlp.add_module("linear1", nn.Linear(num_channels, num_hidden))
        self.mlp.add_module("act1", activation_to_module(activation))
        self.mlp.add_module("linear2", nn.Linear(num_hidden, num_hidden))
        self.mlp.add_module("act2", activation_to_module(activation))
        self.mlp.add_module("linear3", nn.Linear(num_hidden, num_channels))
        self.mlp.add_module("act3", activation_to_module(activation))

    def forward(self, x):
        y = self.mlp(x)
        return x + y


class PatchAutoencoder2d(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            lib_size: int,
            patch_size: Union[int, Tuple[int, int]],
            mlp_blocks: int = 2,
            mlp_cells: Optional[int] = None,
            mlp_activation: Union[None, str, Callable] = "relu",
            output_activation: Union[None, str, Callable] = "sigmoid",
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = code_size
        self.lib_size = lib_size
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.patch_size = patch_size

        self.grid_shape = (shape[-2] // patch_size[-2], shape[-1] // patch_size[-1])
        grid_size = lib_size * math.prod(self.grid_shape)
        if mlp_cells is None:
            mlp_cells = grid_size

        self.encoder = nn.Sequential()
        self.encoder.add_module("patch", nn.Conv2d(shape[0], lib_size, kernel_size=patch_size, stride=patch_size))
        self.encoder.add_module("flatten", nn.Flatten(-3))
        for i in range(mlp_blocks):
            self.encoder.add_module(f"block{i+1}", ResMLP(grid_size, mlp_cells, activation=mlp_activation))
        #self.encoder.add_module("d", DebugLayer())
        self.encoder.add_module(f"proj", nn.Linear(grid_size, code_size))

        print(self.grid_shape, grid_size)
        self.decoder = nn.Sequential()
        self.decoder.add_module(f"proj", nn.Linear(code_size, grid_size))
        for i in range(mlp_blocks):
            self.decoder.add_module(f"block{i+1}", ResMLP(grid_size, mlp_cells, activation=mlp_activation))
        self.decoder.add_module("unflatten", Reshape((lib_size, *self.grid_shape)))
        self.decoder.add_module("patch", nn.ConvTranspose2d(lib_size, shape[0], kernel_size=patch_size, stride=patch_size))
        if output_activation is not None:
            self.decoder.add_module("act_out", activation_to_module(output_activation))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def extra_repr(self):
        return f"shape={self.shape}, code_size={self.code_size}, lib_size={self.lib_size}, patch_size={self.patch_size}"
