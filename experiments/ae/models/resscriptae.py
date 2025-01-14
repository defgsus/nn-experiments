import math
import re
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module


class ResConv(nn.Module):

    def __init__(
            self,
            num_in: int,
            num_out: int,
            kernel_size: int = 3,
            activation: Union[None, str, Callable] = None,
            groups: int = 1,
            residual: str = "add",  # "add", "map"
            transpose: bool = False,
    ):
        super().__init__()

        conv_class = nn.Conv2d
        if transpose:
            conv_class = nn.ConvTranspose2d

        self.residual = None
        self.residual_mode = residual
        self.transpose = transpose
        self.num_in = num_in
        self.num_out = num_out
        if num_in != num_out:
            if self.residual_mode == "map":
                self.residual = nn.Conv2d(num_in, num_out, 1, bias=False, groups=groups)

        padding = int(math.floor(kernel_size / 2))
        self.conv = conv_class(num_in, num_out, kernel_size, padding=padding, groups=groups)
        self.act = activation_to_module(activation)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        if self.residual is not None:
            r = self.residual(r)

        y = self.conv(x)
        if self.act is not None:
            y = self.act(y)

        if self.residual_mode == "add":
            if self.num_in < self.num_out:
                idx = 0
                while idx < self.num_out:
                    idx2 = idx + self.num_in
                    rsize = self.num_out - idx
                    y[..., idx: idx2, :, :] = y[..., idx: idx2, :, :] + r[..., :rsize, :, :]
                    idx += self.num_in
            elif self.num_in > self.num_out:
                idx = 0
                while idx < self.num_in:
                    idx2 = idx + self.num_out
                    ysize = self.num_in - idx
                    y[..., :ysize, :, :] = y[..., :ysize, :, :] + r[..., idx: idx2, :, :]
                    idx += self.num_out
        else:
            y = y + r
        return y


class ResidualScriptedAE(nn.Module):

    COMMANDS = {
        "ch": re.compile(r"ch\s*([*/=])\s*(\d+)"),
        "down": re.compile(r"down(\d)"),
    }

    def __init__(
            self,
            channels: int,
            script: str,
            kernel_size: int = 1,
            padding: int = 0,
            activation: Union[None, str, Callable] = None,
            residual: str = "add",
            groups: int = 1,
    ):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.script = "|".join(filter(bool, (l.strip() for l in script.splitlines())))

        ch = channels
        for line in self.script.split("|"):
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match, args = None, None
            for cmd, regex in self.COMMANDS.items():
                match = regex.match(line)
                if match:
                    args = list(match.groups())
                    break

            if match is None:
                raise SyntaxError(f"Could not parse line `{line}`")

            if cmd == "ch":
                if args[0] == "=":
                    new_ch = int(eval(args[1]))
                else:
                    new_ch = int(eval(f"{ch} {args[0]} {args[1]}"))
                groups_param = 1
                if ch / groups == ch // groups and new_ch / groups == new_ch // groups:
                    groups_param=groups
                self.encoder.append(ResConv(ch, new_ch, kernel_size=kernel_size, activation=activation, groups=groups_param, residual=residual))
                self.decoder.insert(0, ResConv(new_ch, ch, kernel_size=kernel_size, activation=activation, groups=groups_param, residual=residual, transpose=True))
                ch = new_ch

            elif cmd == "down":
                self.encoder.append(nn.PixelUnshuffle(2))
                self.decoder.insert(0, nn.PixelShuffle(2))
                ch = ch * 4

        self.decoder.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def extra_repr(self) -> str:
        return f"script={repr(self.script)}"

    def debug_forward(self, x: torch.Tensor, file=None) -> torch.Tensor:
        print(f"encoding {x.shape}", file=file)
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            print(f"{type(layer).__name__:20} -> {math.prod(x.shape):10} = {tuple(x.shape)}", file=file)
        print(f"decoding {x.shape}", file=file)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            print(f"{type(layer).__name__:20} -> {math.prod(x.shape):10} = {tuple(x.shape)}", file=file)
        return x


if __name__ == "__main__":

    # use it like this:
    ResidualScriptedAE(
        channels=3,
        kernel_size=3,
        script="""
            ch=32
            ch*2
            ch*2
            down
            ch/4
            down
            ch/4
            down
            ch/4
            ch/2
            ch/2
            ch/2
        """
    )
