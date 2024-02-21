"""
Recursively Branched Deconvolutional Network

Inspired by:

Generalized Deep Image to Image Regression
Venkataraman Santhanam, Vlad I. Morariu, Larry S. Davis
https://arxiv.org/abs/1612.03268


Unfortunately, the numbers for the max-pools and branches are not specified in the paper.
Only the main branch (B0) is specified as ks=9, ch=64, hidden-ks=3, layers=9

Also the concatenation is not specified. Currently, this code concats the 64 channels of the
first conv with the 64 channels of the sub-branch which gives 128 hidden channels.
"""
import unittest
from typing import Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_callable, activation_to_module


class RBDNConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            activation: Union[None, str, Callable],
            batch_norm: bool,
            transposed: bool = False,
    ):
        super().__init__()

        self.conv = (nn.ConvTranspose2d if transposed else nn.Conv2d)(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        )
        self.act = activation_to_module(activation)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(
            self,
            x: torch.Tensor,
            output_size: Union[None, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = self.conv(x)
        if output_size is not None and tuple(x.shape[-2:]) != output_size:
            x = F.pad(x, (0, output_size[-1] - x.shape[-1], 0, output_size[-2] - x.shape[-2]))

        if self.act:
            x = self.act(x)

        if hasattr(self, "bn"):
            x = self.bn(x)

        return x


class RBDNBranch(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            num_hidden_layers: int = 1,

            conv_kernel_size: int = 3,
            conv_stride: int = 1,
            conv_padding: int = 0,
            pool_kernel_size: int = 3,
            pool_stride: int = 1,

            hidden_kernel_size: int = 3,
            hidden_stride: int = 1,
            hidden_padding: int = 1,

            batch_norm: bool = True,
            batch_norm_last_layer: bool = False,
            activation: Union[None, str, Callable] = "relu",
            activation_last_layer: Union[None, str, Callable] = "relu",
            sub_branch: Union[None, "RBDNBranch"] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        actual_hidden_channels = hidden_channels
        if sub_branch is not None:
            actual_hidden_channels += sub_branch.out_channels

        self.conv_in = RBDNConv(
            in_channels, hidden_channels, conv_kernel_size, stride=conv_stride, padding=conv_padding,
            activation=activation, batch_norm=batch_norm
        )

        self.sub_branch = sub_branch

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, return_indices=True)

        self.hidden = nn.Sequential()
        for i in range(num_hidden_layers):
            self.hidden.add_module(f"conv_{i+1}", RBDNConv(
                actual_hidden_channels, actual_hidden_channels, hidden_kernel_size, stride=hidden_stride, padding=hidden_padding,
                activation=activation, batch_norm=batch_norm
            ))

        self.unpool = nn.MaxUnpool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.conv_out = RBDNConv(
            actual_hidden_channels, out_channels, conv_kernel_size, stride=conv_stride, padding=conv_padding,
            activation=activation_last_layer, batch_norm=batch_norm_last_layer,
            transposed=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(input)

        if self.sub_branch is not None:
            sub_x = self.sub_branch(x)
            x = torch.concat([x, sub_x], dim=-3)

        unpooled_size = x.shape[-2:]
        x, indices = self.pool(x)

        # print("hidden:", x.shape)
        x = self.hidden(x)

        x = self.unpool(x, indices, output_size=unpooled_size)
        x = self.conv_out(x, output_size=input.shape[-2:])

        return x

    @torch.no_grad()
    def get_inner_shape(self, shape: Tuple[int, int, int]) -> dict:
        x = self.conv_in(torch.zeros(1, *shape))

        if self.sub_branch is not None:
            branch_shape = self.sub_branch.get_inner_shape(x.shape[-3:])
            sub_x = self.sub_branch(x)
            x = torch.concat([x, sub_x], dim=-3)
        else:
            branch_shape = None

        x, indices = self.pool(x)

        ret = {"shape": x.shape[-3:], "branch": branch_shape}
        return ret


class RBDN(nn.Module):
    """
    Recursively Branched Deconvolutional Network

    See:

        Generalized Deep Image to Image Regression
        Venkataraman Santhanam, Vlad I. Morariu, Larry S. Davis
        https://arxiv.org/abs/1612.03268
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            num_branches: int,
            num_hidden_layers: int = 1,

            conv_kernel_size: int = 3,
            conv_stride: int = 1,
            conv_padding: int = 0,
            pool_kernel_size: int = 3,
            pool_stride: int = 1,

            branch_conv_kernel_size: int = 3,
            branch_conv_stride: int = 2,
            branch_conv_padding: int = 1,
            branch_pool_kernel_size: int = 3,
            branch_pool_stride: int = 1,
            branch_num_hidden_layers: int = 1,

            hidden_kernel_size: int = 3,
            hidden_stride: int = 1,
            hidden_padding: int = 1,

            batch_norm: bool = True,
            batch_norm_last_layer: bool = False,
            activation: Union[None, str, Callable] = "relu",
            activaton_last_layer: Union[None, str, Callable] = "sigmoid",
    ):
        super().__init__()

        branches = None
        for i in range(num_branches):
            branches = RBDNBranch(
                hidden_channels, hidden_channels, hidden_channels,
                conv_kernel_size=branch_conv_kernel_size, conv_stride=branch_conv_stride, conv_padding=branch_conv_padding,
                pool_kernel_size=branch_pool_kernel_size, pool_stride=branch_pool_stride,
                hidden_kernel_size=hidden_kernel_size, hidden_stride=hidden_stride, hidden_padding=hidden_padding,
                activation=activation, batch_norm=batch_norm,
                num_hidden_layers=branch_num_hidden_layers,
                sub_branch=branches,
            )

        self.branches = RBDNBranch(
            in_channels, out_channels, hidden_channels,
            conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, conv_padding=conv_padding,
            pool_kernel_size=pool_kernel_size, pool_stride=pool_stride,
            hidden_kernel_size=hidden_kernel_size, hidden_stride=hidden_stride, hidden_padding=hidden_padding,
            activation=activation,
            batch_norm=batch_norm,
            num_hidden_layers=num_hidden_layers,
            sub_branch=branches,
            batch_norm_last_layer=batch_norm_last_layer,
            activation_last_layer=activaton_last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branches(x)

    def get_inner_shape(self, shape: Tuple[int, int, int]) -> dict:
        return self.branches.get_inner_shape(shape)


class TestRBDN(unittest.TestCase):

    @torch.no_grad()
    def test_rbdn_branch(self):
        for shape in (
                (3, 64, 64),
                (3, 63, 65),
        ):
            for conv_stride in range(1, 8):
                for conv_kernel_size in range(1, 8):
                    for pool_stride in range(1, 8):
                        for pool_kernel_size in range(1, 8):

                            msg = (
                                f"shape={shape}"
                                f", conv_stride={conv_stride}, conv_kernel_size={conv_kernel_size}"
                                f", pool_stride={pool_stride}, pool_kernel_size={pool_kernel_size}"
                            )
                            model = RBDN(
                                in_channels=shape[0],
                                hidden_channels=10,
                                out_channels=shape[0],
                                num_branches=1,
                                conv_stride=conv_stride,
                                conv_kernel_size=conv_kernel_size,
                                pool_stride=pool_stride,
                                pool_kernel_size=pool_kernel_size,
                            ).eval()

                            self.assertEqual(
                                torch.Size(shape),
                                model(torch.zeros(1, *shape)).shape[-3:],
                                msg
                            )
