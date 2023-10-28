import math
from typing import Tuple, Optional, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block2d import Conv2dBlock


class ConvAutoEncoder(torch.nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int] = None,
            kernel_size: Union[int, Iterable[int]] = 5,
            code_size: int = 128,
            act_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
            batch_norm: bool = False,
            bias: bool = True,
            linear_bias: bool = True,
            act_last_layer: bool = True,
            space_to_depth: bool = False,
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
            bias=bias,
            space_to_depth=space_to_depth,
        )
        conv_shape = encoder_block.get_output_shape(self.shape)
        self.encoder = torch.nn.Sequential(
            encoder_block,
            nn.Flatten(),
            nn.Linear(math.prod(conv_shape), code_size, bias=linear_bias),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(code_size, math.prod(conv_shape), bias=linear_bias),
            nn.Unflatten(1, conv_shape),
            encoder_block.create_transposed(
                act_last_layer=act_last_layer,
            )
        )

    def conv_parameters(self):
        return list(self.encoder[:-1].parameters()) + list(self.decoder[1:].parameters())

    def add_layer(self, channels: int, kernel_size: Optional[int] = None, bias: bool = True):
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        self.encoder[0].add_output_layer(channels=channels, kernel_size=kernel_size, bias=bias)
        self.decoder[-1].add_input_layer(channels=channels, kernel_size=kernel_size, bias=bias, transpose=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encode(x)
        return self.decode(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.decoder.forward(x)
        return y.reshape(-1, *self.shape)

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
