from typing import Tuple, Optional, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):

    def __init__(
            self,
            channels: Iterable[int],
            kernel_size: int = 5,
            stride: int = 1,
            pool_kernel_size: int = 0,
            pool_type: str = "max",  # "max", "average"
            act_fn: Optional[nn.Module] = None,
            act_last_layer: bool = True,
            bias: bool = True,
            transpose: bool = False,
            batch_norm: bool = False,
    ):
        super().__init__()
        self.channels = list(channels)
        self._act_fn = act_fn
        assert len(self.channels) >= 2, f"Got: {self.channels}"

        self.layers = nn.Sequential()

        if batch_norm:
            self.layers.append(
                nn.BatchNorm1d(self.channels[0])
            )

        for i, (channels, next_channels) in enumerate(zip(self.channels, self.channels[1:])):
            self.layers.append(
                self._create_layer(
                    in_channels=channels,
                    out_channels=next_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    transpose=transpose,
                )
            )
            if pool_kernel_size and i == len(self.channels) - 2:
                klass = {
                    "max": nn.MaxPool1d,
                    "average": nn.AvgPool1d,
                }[pool_type]
                self.layers.append(
                    klass(pool_kernel_size)
                )
            if self._act_fn and (act_last_layer or i + 2 < len(self.channels)):
                self.layers.append(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers.forward(x)
        return y

    @torch.no_grad()
    def get_output_shape(
            self,
            shape: Union[int, Tuple[int], Tuple[int, int]],
    ) -> Tuple[int, int]:
        if isinstance(shape, int):
            shape = (1, shape)
        elif len(shape) == 1:
            shape = (self.channels[0], *shape)
        x = torch.zeros(1, *shape)
        y = self.forward(x)
        return tuple(y.shape[-2:])

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
        return (nn.ConvTranspose1d if transpose else nn.Conv1d)(
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
                if weight.ndim == 3:
                    for wchan in weight[:64]:
                        for w in wchan[:3]:
                            images.append(w.unsqueeze(0))
        return images
