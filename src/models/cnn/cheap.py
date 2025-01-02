from typing import Union, Type

import torch
import torch.nn as nn


class CheapConvBase(nn.Module):
    """
    Actually called *Depthwise Separable Convolution*
    originally used in *Xception* network.

    see https://paperswithcode.com/method/depthwise-convolution
    """
    def __init__(
            self,
            conv_class: Type[nn.Module],
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Union[int, str] = 0,
            dilation: int = 1,
            bias: bool = False,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._bias = bias

        self.depth_conv = conv_class(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.point_conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def extra_repr(self) -> str:
        text = f"{self._in_channels}, {self._out_channels}, kernel_size={self._kernel_size}"
        if self._stride != 1:
            text = f"{text}, stride={self._stride}"
        if self._padding != 0:
            text = f"{text}, padding={self._padding}"
        if self._dilation != 1:
            text = f"{text}, dilation={self._dilation}"
        if self._bias is not True:
            text = f"{text}, bias={self._bias}"
        return text

    @property
    def weight(self):
        return self.depth_conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class CheapConv1d(CheapConvBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Union[int, str] = 0,
            dilation: int = 1,
            bias: bool = False,
    ):
        super().__init__(
            conv_class=nn.Conv1d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )


class CheapConvTranspose1d(CheapConvBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Union[int, str] = 0,
            dilation: int = 1,
            bias: bool = False,
    ):
        super().__init__(
            conv_class=nn.ConvTranspose1d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )


class CheapConv2d(CheapConvBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Union[int, str] = 0,
            dilation: int = 1,
            bias: bool = False,
    ):
        super().__init__(
            conv_class=nn.Conv2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )


class CheapConvTranspose2d(CheapConvBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Union[int, str] = 0,
            dilation: int = 1,
            bias: bool = False,
    ):
        super().__init__(
            conv_class=nn.ConvTranspose2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
