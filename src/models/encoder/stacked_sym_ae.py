import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import *


class SymmetricLinear(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: Union[None, str, nn.Module] = None,
            symmetric: bool = True,
            #dropout: float = 0.,
            #batch_norm: bool = False,
    ):
        super().__init__()
        self.activation = activation_to_callable(activation)
        self.linear = nn.Linear(in_channels, out_channels)
        if symmetric:
            self.bias_out = nn.Parameter(
                torch.randn(in_channels) * self.linear.bias.std()
            )
        else:
            self.linear_out = nn.Linear(out_channels, in_channels)

    def forward(self, x, transpose: bool = False):
        if not transpose:
            y = self.linear(x)
            if self.activation:
                y = self.activation(y)
        else:
            if hasattr(self, "linear_out"):
                y = self.linear_out(x)
            else:
                y = F.linear(x, self.linear.weight.T, self.bias_out)
            if self.activation:
                y = self.activation(y)
        return y


class SymmetricConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            activation: Union[None, str, nn.Module] = "leaky_relu",
            symmetric: bool = True,
            #space_to_depth: bool = False,
            #dropout: float = 0.,
            #batch_norm: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.activation = activation_to_callable(activation)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv_out = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, output_padding=stride - 1)
        # make weights symmetric by copying whole parameter
        if symmetric:
            self.conv_out.weight = self.conv_in.weight

    def forward(self, x, transpose: bool = False):
        if not transpose:
            y = self.conv_in(x)
            if self.activation:
                y = self.activation(y)
        else:
            y = self.conv_out(x)
            if self.activation:
                y = self.activation(y)
        return y


class StackedSymmetricAutoencoderConv2d(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            kernel_size: Union[int, Iterable[int]] = 3,
            stride: Union[int, Iterable[int]] = 1,
            groups: int = 1,
            channels: Iterable[int] = (16, 32),
            activation: Union[None, str, nn.Module] = "leaky_relu",
            symmetric: bool = True,
            space_to_depth: bool = False,
            dropout: float = 0.,
            batch_norm: bool = False,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = code_size
        self.channels = tuple(channels)
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride if isinstance(stride, int) else tuple(stride)
        self.groups = groups
        self.activation = activation_to_callable(activation)
        self.layer_index = len(self.channels) + 1

        channels = [self.shape[0], *self.channels]
        self.layers = nn.Sequential()
        for idx, ch in enumerate(channels[:-1]):
            ch_in = ch
            ch_out = channels[idx + 1]
            if isinstance(self.kernel_size, int):
                ks = self.kernel_size
            else:
                ks = self.kernel_size[idx]
            if isinstance(self.stride, int):
                stride = self.stride
            else:
                stride = self.stride[idx]

            self.layers.add_module(f"conv_{idx + 1}", SymmetricConv2d(
                ch_in, ch_out,
                kernel_size=ks,
                stride=stride,
                activation=activation,
                symmetric=symmetric,
            ))

        self.conv_shapes = []
        with torch.no_grad():
            data = torch.zeros(1, *self.shape)
            for conv in self.layers:
                data = conv(data)
                self.conv_shapes.append(data.shape[-3:])
        # print(self.conv_shapes)

        self.layers.add_module("linear", SymmetricLinear(
            math.prod(self.conv_shapes[-1]), self.code_size,
            symmetric=symmetric,
        ))

    def forward(self, x):
        y = self.encode(x)
        # print(f"code: {y.shape}")
        x = self.decode(y)
        return x

    def encode(self, x):
        for idx, layer in zip(range(self.layer_index + 1), self.layers):
            if idx == len(self.layers) - 1:
                x = x.flatten(-3)
            x = layer(x)
        if self.layer_index < len(self.layers) - 1:
            x = x.flatten(-3)
        return x

    def decode(self, x):
        if self.layer_index < len(self.conv_shapes) - 1:
            shape = self.conv_shapes[self.layer_index]
            x = x.view(*x.shape[:-1], *shape)

        for idx, layer in reversed(list(zip(range(self.layer_index + 1), self.layers))):
            if idx == len(self.conv_shapes) - 1:
                shape = self.conv_shapes[idx]
                x = x.view(*x.shape[:-1], *shape)
            x = layer(x, transpose=True)
        return x
