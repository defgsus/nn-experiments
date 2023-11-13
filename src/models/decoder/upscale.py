import math
from collections import OrderedDict
from typing import Optional, Tuple, Union, Iterable, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_module
from src.models.transform import Reshape


class ConvUpscaleDecoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            num_layers: int = 4,
            num_hidden: int = 64,
            upscale_every: int = 3,
            activation: Union[None, str, Callable] = "relu",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        def _is_upscale(i):
            return (i + 1) % upscale_every == 0

        # calculate shape at start of de-convolution
        s = list(shape[-2:])
        for i in range(num_layers - 1, -1, -1):
            if _is_upscale(i):
                for j in range(2):
                    if s[j] % 2 != 0:
                        raise ValueError(f"Upscaling in layer {i+1} requires scale divisible by 2, got {s}")
                    s[j] //= 2

            for j in range(2):
                s[j] -= int(math.ceil(kernel_size[j] / 2))
            for j in range(2):
                if s[j] <= 0:
                    raise ValueError(f"Convolution in layer {i+1} requires scale > 0, got {s}")

        start_shape = (num_hidden, *s)

        self.layers = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(code_size, math.prod(start_shape))),
            ("reshape", Reshape(start_shape)),
        ]))
        if activation is not None:
            self.layers.add_module("act_0", activation_to_module(activation))

        ch_in = start_shape[0]
        ch_out = num_hidden
        for i in range(num_layers):

            self.layers.add_module(f"conv_{i + 1}", nn.ConvTranspose2d(
                ch_in,
                ch_out * 4 if _is_upscale(i) else ch_out,
                kernel_size
            ))
            if activation is not None:
                self.layers.add_module(f"act_{i + 1}", activation_to_module(activation))
            ch_in = ch_out

            if _is_upscale(i):
                self.layers.add_module(f"up_{i // upscale_every + 1}", nn.PixelShuffle(2))

        self.layers.add_module(f"conv_out", nn.ConvTranspose2d(ch_in, shape[0], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
