import math
from typing import Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import Conv2dBlock


class ConvAutoEncoder(torch.nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int] = None,
            kernel_size: int = 5,
            code_size: int = 128,
            act_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
            batch_norm: bool = False,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = code_size
        self._act_fn = act_fn

        assert self.shape[-2] == self.shape[-1], self.shape

        self.kernel_size = kernel_size

        self.channels = [shape[0]] + list(channels)
        encoder_block = Conv2dBlock(
            channels=self.channels,
            kernel_size=self.kernel_size,
            act_fn=self._act_fn,
            batch_norm=batch_norm,
            #act_last_layer=True,
        )
        conv_shape = encoder_block.get_output_shape(self.shape)
        self.encoder = torch.nn.Sequential(
            encoder_block,
            nn.Flatten(),
            nn.Linear(math.prod(conv_shape), code_size),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(code_size, math.prod(conv_shape)),
            nn.Unflatten(1, conv_shape),
            Conv2dBlock(
                channels=list(reversed(self.channels)),
                kernel_size=self.kernel_size,
                act_fn=self._act_fn,
                batch_norm=batch_norm,
                transpose=True,
                # act_last_layer=True,
            )
        )

    def add_layer(self, channels: int, kernel_size: Optional[int] = None, bias: bool = True):
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        self.encoder[0].add_output_layer(channels=channels, kernel_size=kernel_size, bias=bias)
        self.decoder[-1].add_input_layer(channels=channels, kernel_size=kernel_size, bias=bias, transpose=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encode(x)
        return self.decode(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder.forward(x).reshape(-1, *self.shape)
        return x

    def weight_images(self, **kwargs):
        images = []
        for layers in (self.encoder[0].layers, self.decoder[-1].layers):
            for layer in layers:
                if hasattr(layer, "weight"):
                    weight = layer.weight
                    if weight.ndim == 4:
                        for w in weight[:5]:
                            images.append(w[0])
                        for w in weight[0, 1:6]:
                            images.append(w)
        return images
