import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module


class ScriptAE(nn.Module):

    def __init__(
            self,
            script: str,
            verbose: bool = False,
    ):
        super().__init__()
        self.script = script
        self.verbose = verbose
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for cmd in self.script.splitlines():
            if "#" in cmd:
                cmd = cmd[:cmd.index("#")]
            cmd = cmd.strip()
            if not cmd:
                continue

            if "(" not in cmd or ")" not in cmd:
                raise SyntaxError(f"Can not parse command '{cmd}'")

            args = cmd[cmd.index("(")+1:-1].strip()
            args = [a.strip() for a in args.split(",")]
            cmd = cmd[:cmd.index("(")]

            for i, a in enumerate(args):
                try:
                    args[i] = int(a)
                except:
                    pass

            if cmd == "ps":
                self.encoder.append(nn.PixelUnshuffle(*args))
                self.decoder.insert(0, nn.PixelShuffle(*args))

            elif cmd == "bn":
                self.encoder.append(nn.BatchNorm2d(*args))
                self.decoder.insert(0, nn.BatchNorm2d(*args))

            elif cmd == "conv":
                self.encoder.append(nn.Conv2d(*args))
                self.decoder.insert(0, nn.ConvTranspose2d(args[1], args[0], *args[2:]))

            else:
                raise SyntaxError(f"Unknown cmd '{cmd}'")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print("encode")
        y = x
        for b in self.encoder:
            if self.verbose:
                bs = str(b).replace('\n', ' ')
                print(f"{y.shape} -> {bs}")
            y = b(y)
        if self.verbose:
            print("->", y.shape)
        return y

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print("decode")
        y = x
        for b in self.decoder:
            if self.verbose:
                bs = str(b).replace('\n', ' ')
                print(f"{y.shape} -> {bs}")
            y = b(y)
        if self.verbose:
            print("->", y.shape)
        return y

    def forward(self, x):
        return self.decode(self.encode(x))

if __name__ == "__main__":

    # use it like this:
    ScriptAE(
        verbose=True,
        script="""
            conv(3,16,3,1,1)
            ps(2)
            conv(64,32,3,1,1)
            bn(32)
            ps(2)
            conv(128,64,3,1,1)
            bn(64)
            conv(64,32,3,1,1)
            conv(32,8,3,1,1)
        """,
    )
