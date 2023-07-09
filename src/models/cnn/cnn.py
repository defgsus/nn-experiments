from typing import Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):

    def __init__(
            self,
            channels: Iterable[int],
            kernel_size: int = 5,
            act_fn: Optional[nn.Module] = None,
            bias: bool = True,
            transpose: bool = False,
    ):
        super().__init__()
        self.channels = list(channels)
        self._act_fn = act_fn
        assert len(self.channels) >= 2, f"Got: {channels}"

        self.layers = nn.Sequential()

        for channels, next_channels in zip(self.channels, self.channels[1:]):
            self.layers.append(
                self._create_layer(
                    in_channels=channels,
                    out_channels=next_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                    transpose=transpose,
                )
            )
            if self._act_fn:
                self.layers.append(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)

    def get_output_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        with torch.no_grad():
            x = torch.zeros(self.channels[0], shape[0], shape[1])
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
            bias: bool = True,
            transpose: bool = False,
    ) -> nn.Module:
        return (nn.ConvTranspose2d if transpose else nn.Conv2d)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )