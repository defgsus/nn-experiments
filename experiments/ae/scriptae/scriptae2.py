import math
import re
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module
from src.models.attention import Attention1d


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels: int = 0):
        super().__init__()
        self._channels = channels
        if channels:
            self.conv = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        else:
            self.conv = None
        self.attn = Attention1d()

    def extra_repr(self) -> str:
        return f"channels={self._channels}"

    def forward(self, x):
        shape = x.shape
        if self.conv is not None:
            qk = self.conv(x)
            x = x.flatten(-2)
            qk = qk.flatten(-2)
            q, k = torch.split(qk, x.shape[-2], dim=-2)
        else:
            x = x.flatten(-2)
            q, k = x, x
        x = self.attn(q, k, x)
        return x.reshape(shape)


class ScriptAELayer(nn.Module):

    def __init__(
            self,
            num_in: int,
            num_out: int,
            kernel_size: int = 3,
            padding: int = 0,
            activation: Union[None, str, Callable] = None,
            batch_norm: bool = False,
            transpose: bool = False,
    ):
        super().__init__()

        conv_class = nn.Conv2d
        if transpose:
            conv_class = nn.ConvTranspose2d

        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_in)
        self.conv = conv_class(num_in, num_out, kernel_size, padding=padding)
        self.act = activation_to_module(activation)

    def forward(self, x):
        original_x = x

        if self.bn is not None:
            x = self.bn(x)

        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)

        if x.shape == original_x.shape:
            x = x + original_x

        return x


class ScriptedAE(nn.Module):

    COMMANDS = {
        # conv: ch*2, ch/2, ch=32
        "ch": re.compile(r"ch\s*([*/=])\s*(\d+)"),
        # pixel-unshuffle: down2
        "down": re.compile(r"down(\d)"),
        # attention
        "aqk": re.compile(r"aqk"),
        "a": re.compile(r"a"),
    }

    def __init__(
            self,
            channels: int,
            script: str,
            kernel_size: int = 1,
            padding: Optional[int] = None,
            activation: Union[None, str, Callable] = None,
            final_encoder_activation: Union[None, str, Callable] = None,
            final_decoder_activation: Union[None, str, Callable] = None,
            batch_norm: bool = False,
    ):
        assert kernel_size >= 1, f"Got kernel_size={kernel_size}"

        if padding is None:
            padding = (kernel_size - 1) // 2

        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.script = "|".join(filter(bool, (l.strip() for l in script.splitlines())))

        ch = channels
        lines = self.script.split("|")
        for line_idx, line in enumerate(lines):
            is_last_encoder_channel = line_idx == len(lines) - 1
            is_last_decoder_channel = line_idx == 0

            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            add_encoder = True
            add_decoder = True
            if line.startswith("e:"):
                line = line[2:]
                add_decoder = False
            if line.startswith("d:"):
                line = line[2:]
                add_encoder = False

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

                if add_encoder:
                    self.encoder.append(ScriptAELayer(
                        num_in=ch,
                        num_out=new_ch,
                        kernel_size=kernel_size,
                        padding=padding,
                        activation=final_encoder_activation if is_last_encoder_channel else activation,
                        batch_norm=batch_norm and not is_last_encoder_channel,
                    ))
                if add_decoder:
                    self.decoder.insert(0, ScriptAELayer(
                        num_in=new_ch,
                        num_out=ch,
                        kernel_size=kernel_size,
                        padding=padding,
                        activation=final_decoder_activation if is_last_decoder_channel else activation,
                        batch_norm=batch_norm and not is_last_decoder_channel,
                        transpose=True,
                    ))
                ch = new_ch

            elif cmd == "down":
                factor = int(args[0])
                if add_encoder:
                    self.encoder.append(nn.PixelUnshuffle(factor))
                if add_decoder:
                    self.decoder.insert(0, nn.PixelShuffle(factor))
                ch = ch * factor ** 2

            elif cmd == "a":
                if add_encoder:
                    self.encoder.append(SelfAttentionLayer())
                if add_decoder:
                    self.decoder.insert(0, SelfAttentionLayer())

            elif cmd == "aqk":
                if add_encoder:
                    self.encoder.append(SelfAttentionLayer(ch))
                if add_decoder:
                    self.decoder.insert(0, SelfAttentionLayer(ch))

        # self.decoder.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def extra_repr(self) -> str:
        return f"script={repr(self.script)}"


if __name__ == "__main__":

    # use it like this:
    ScriptedAE(
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
