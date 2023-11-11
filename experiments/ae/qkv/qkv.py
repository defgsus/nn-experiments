import math
from collections import OrderedDict
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size: int = 5):
        super().__init__()
        num_hid = max(2, num_in)
        self.q_conv = nn.Conv2d(num_in, num_hid, kernel_size)
        self.k_conv = nn.Conv2d(num_in, num_hid, kernel_size)
        self.v_conv = nn.Conv2d(num_in, num_hid, kernel_size, padding=kernel_size // 2)
        self.s_conv = nn.Conv2d(num_hid * num_hid, num_hid, 1)
        self.out_conv = nn.Conv2d(num_hid, num_out, 1)
        self.residual = nn.Identity() if num_in == num_out else nn.Conv2d(num_in, num_out, 1)

    def forward(self, x):
        assert x.ndim == 4, x.shape
        bs = x.shape[0]
        q = self.q_conv(x).flatten(-2)                # BxCx(HxW)
        k = self.k_conv(x).flatten(-2)
        v = self.v_conv(x)
        s = F.softmax(q @ k.permute(0, 2, 1), dim=1)  # BxCxC
        s = self.s_conv(s.view(bs, -1, 1, 1))         # BxCx1x1
        y = v * s                                     # BxCxHxW
        y = self.out_conv(y)                          # BxCxHxW
        y = F.relu(self.residual(x) + y)
        return y


class Encoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            num_out: int,
            channels_ks: Tuple[Tuple[int, int], ...] = ((16, 3), (32, 5), (32, 7)),
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.num_out = num_out
        self.channels_ks = tuple(channels_ks)

        self.blocks = nn.Sequential(OrderedDict([
            (
                f"block_{i + 1}",
                EncoderBlock(
                    num_in=self.shape[0] if i == 0 else self.channels_ks[i - 1][0],
                    num_out=ch_out,
                    kernel_size=ks,
                )
            )
            for i, (ch_out, ks) in enumerate(self.channels_ks)
        ]))
        with torch.no_grad():
            self._conv_shape = self.blocks(torch.zeros(1, *self.shape)).shape

        self.w_out = nn.Linear(math.prod(self._conv_shape), self.num_out)

    def forward(self, x):
        assert x.ndim == 4, x.shape
        conv = self.blocks(x)
        code = self.w_out(conv.flatten(1))
        return code

    def extra_repr(self):
        return (
            f"shape={self.shape}, num_out={self.num_out},\nchannels_ks={self.channels_ks}"
            f",\n_conv_shape={self._conv_shape}"
        )
