from typing import Tuple, Optional, Iterable, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_module
from .spacedepth import SpaceToDepth


class Conv2dBlock(nn.Module):

    def __init__(
            self,
            channels: Iterable[int],
            kernel_size: Union[int, Iterable[int]] = 5,
            stride: int = 1,
            pool_kernel_size: int = 0,
            pool_type: str = "max",  # "max", "average"
            act_fn: Optional[nn.Module] = None,
            act_last_layer: Union[None, bool, str, nn.Module, Callable] = True,
            bias: bool = True,
            transpose: bool = False,
            batch_norm: bool = False,
            space_to_depth: bool = False,
            dropout: float = 0.,
    ):
        super().__init__() #channels, kernel_size, stride, pool_kernel_size, pool_type, act_fn, act_last_layer, bias, transpose, batch_norm, space_to_depth)
        self.channels = list(channels)
        if len(self.channels) < 2:
            raise ValueError(f"Expected `channels` to be of length >= 2, got: {self.channels}")

        num_layers = len(self.channels) - 1

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size for _ in range(num_layers)]
        else:
            self.kernel_size = list(kernel_size)
            if len(self.kernel_size) != num_layers:
                raise ValueError(f"Expected `kernel_size` to have {num_layers} elements, got {self.kernel_size}")

        self.stride = stride
        self._bias = bias
        self._batch_norm = batch_norm
        self._space_to_depth = space_to_depth
        self._act_fn = act_fn
        self._pool_kernel_size = pool_kernel_size
        self._pool_type = pool_type
        self._act_last_layer = act_last_layer
        self._transpose = transpose
        self._dropout = dropout

        self.layers = nn.Sequential()

        in_channel_mult = 1
        out_channel_mult = 1
        for i, (in_channels, out_channels, kernel_size) in enumerate(
                zip(self.channels, self.channels[1:], self.kernel_size)
        ):
            is_last_layer = i == len(self.channels) - 2

            if space_to_depth and transpose:
                self.layers.append(SpaceToDepth(transpose=transpose))
                out_channel_mult = 1
                if i < len(self.channels) - 2:
                    out_channel_mult = 4

            self.layers.append(
                self._create_layer(
                    in_channels=in_channels * in_channel_mult,
                    out_channels=out_channels * out_channel_mult,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    transpose=transpose,
                )
            )
            if batch_norm:
                self.layers.append(
                    nn.BatchNorm2d(out_channels * out_channel_mult)
                )

            if dropout and not is_last_layer:
                self.layers.append(nn.Dropout2d(dropout))

            if space_to_depth and not transpose and i < len(self.channels) - 1:
                self.layers.append(SpaceToDepth(transpose=transpose))
                in_channel_mult = 4

            if pool_kernel_size and is_last_layer:
                klass = {
                    "max": nn.MaxPool2d,
                    "average": nn.AvgPool2d,
                }[pool_type]
                self.layers.append(
                    klass(pool_kernel_size)
                )

            if not is_last_layer:
                if self._act_fn:
                    self.layers.append(self._act_fn)
            else:
                if act_last_layer:
                    act_fn = self._act_fn
                    if not isinstance(act_last_layer, bool):
                        act_fn = activation_to_module(act_last_layer)
                    if act_fn:
                        self.layers.append(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers.forward(x)
        return y

    @torch.no_grad()
    def get_output_shape(
            self,
            shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int, int]:
        if len(shape) == 2:
            shape = (self.channels[0], *shape)
        x = torch.zeros(1, *shape)
        y = self.forward(x)
        return tuple(y.shape[-3:])

    @torch.no_grad()
    def get_input_shape(
            self,
            shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int, int]:
        trans_self = self.create_transposed()
        return trans_self.get_output_shape(shape)

    def create_transposed(
            self,
            act_last_layer: Optional[bool] = None,
    ) -> "Conv2dBlock":
        return self.__class__(
            channels=list(reversed(self.channels)),
            kernel_size=list(reversed(self.kernel_size)),
            stride=self.stride,
            pool_kernel_size=self._pool_kernel_size,
            pool_type=self._pool_type,
            act_fn=self._act_fn,
            act_last_layer=self._act_last_layer if act_last_layer is None else act_last_layer,
            bias=self._bias,
            transpose=not self._transpose,
            batch_norm=self._batch_norm,
            space_to_depth=self._space_to_depth,
            dropout=self._dropout,
        )

    def add_input_layer(
            self,
            channels: int,
            kernel_size: int = 5,
            bias: bool = True,
            transpose: bool = False,
    ):
        self.channels.insert(0, channels)
        self.layers.insert(0, self._create_layer(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=kernel_size,
            bias=bias,
            transpose=transpose,
        ))
        if self._act_fn is not None:
            self.layers.insert(1, self._act_fn)

    def add_output_layer(
            self,
            channels: int,
            kernel_size: int = 5,
            bias: bool = True,
            transpose: bool = False,
    ):
        self.channels.append(channels)
        self.layers.append(self._create_layer(
            in_channels=self.channels[-2],
            out_channels=self.channels[-1],
            kernel_size=kernel_size,
            bias=bias,
            transpose=transpose,
        ))
        if self._act_fn is not None:
            self.layers.append(self._act_fn)

    def _create_layer(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            bias: bool = True,
            transpose: bool = False,
    ) -> nn.Module:
        return (nn.ConvTranspose2d if transpose else nn.Conv2d)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )

    def weight_images(self, **kwargs):
        images = []

        for layer in self.layers:
            if hasattr(layer, "weight"):
                weight = layer.weight
                if weight.ndim == 4:
                    for wchan in weight[:64]:
                        for w in wchan[:3]:
                            images.append(w)
        return images
