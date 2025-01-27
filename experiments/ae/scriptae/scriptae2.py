import math
import re
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_module, normalization_to_module
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


class Conv2dPoly(nn.Module):
    def __init__(
            self,
            order: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            transpose: bool = False,
    ):
        super().__init__()
        self._order = order
        self._in_channels = in_channels

        self.conv_split = nn.Conv2d(in_channels, in_channels * order, kernel_size=1, groups=in_channels)
        conv_class = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv_combine = conv_class(in_channels * order, out_channels, kernel_size, stride, padding)

    def extra_repr(self):
        return f"order={self._order}"

    def forward(self, x):
        y = self.conv_split(x)

        y = torch.concat([
            y[..., i * self._in_channels: (i + 1) * self._in_channels, :, :] ** (i + 1)
            for i in range(self._order)
        ], dim=-3)

        y = self.conv_combine(y)
        return y


class DownLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            factor: int,
    ):
        from src.models.efficientvit.ops import (
            PixelUnshuffleChannelAveragingDownSampleLayer,
            ConvPixelUnshuffleDownSampleLayer,
            ResidualBlock,
        )
        super().__init__()
        self.block = ResidualBlock(
            main=ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, 3, factor),
            shortcut=PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            factor: int,
    ):
        from src.models.efficientvit.ops import (
            ConvPixelShuffleUpSampleLayer,
            ChannelDuplicatingPixelUnshuffleUpSampleLayer,
            ResidualBlock,
        )
        super().__init__()
        self.block = ResidualBlock(
            main=ConvPixelShuffleUpSampleLayer(in_channels, out_channels, 3, factor),
            shortcut=ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScriptAELayer(nn.Module):

    def __init__(
            self,
            num_in: int,
            num_out: int,
            kernel_size: int = 3,
            padding: int = 0,
            activation: Union[None, str, Callable] = None,
            norm: Optional[str] = None,
            poly_order: int = 0,
            skip_conv_layers: int = 0,
            skip_conv_activation: Union[None, str, Callable] = None,
            transpose: bool = False,
    ):
        super().__init__()
        self._poly_order = poly_order
        self._skip_conv_layers = skip_conv_layers

        conv_class = nn.Conv2d
        if transpose:
            conv_class = nn.ConvTranspose2d

        self.norm = None
        if norm:
            self.norm = normalization_to_module(norm, channels=num_in)

        if skip_conv_layers > 0:
            self.skip_conv = nn.Sequential()
            for i in range(skip_conv_layers):
                self.skip_conv.append(
                    conv_class(num_in, num_in, kernel_size, padding=(kernel_size - 1) // 2)
                )
                if (act := activation_to_module(skip_conv_activation)) is not None:
                    self.skip_conv.append(act)

        if poly_order:
            self.conv = Conv2dPoly(poly_order, num_in, num_out, kernel_size, padding=padding, transpose=transpose)
        else:
            self.conv = conv_class(num_in, num_out, kernel_size, padding=padding)

        self.act = activation_to_module(activation)

        with torch.no_grad():
            x = torch.zeros(1, num_in, 64, 64)
            y = self.conv(x)
            self._residual = x.shape == y.shape

    def extra_repr(self) -> str:
        return f"residual={self._residual}, poly_order={self._poly_order}, skip_conv_layers={self._skip_conv_layers}"

    def forward(self, x):
        original_x = x

        if self.norm is not None:
            x = self.norm(x)

        if self._skip_conv_layers > 0:
            x2 = self.skip_conv(x)
            x = x2 + x

        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)

        if x.shape == original_x.shape:
            x = x + original_x

        return x


class VariationalLayer(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 3, kl_loss_weight: float = 1.):
        super().__init__()
        self._channels = channels
        self._kl_loss_weight = kl_loss_weight

        self.distribution = torch.distributions.Normal(0, 1)
        self.last_mu: Optional[torch.Tensor] = None
        self.last_sigma: Optional[torch.Tensor] = None

        self.conv_sigma_mu = nn.Conv2d(channels, channels * 2, kernel_size, padding=(kernel_size - 1) // 2)

    def extra_repr(self) -> str:
        return f"kl_loss_weight={self._kl_loss_weight}"

    def forward(self, x: torch.Tensor):
        # move sampler to GPU
        device = x.device
        if self.distribution.loc.device != device:
            self.distribution.loc = self.distribution.loc.to(device)
            self.distribution.scale = self.distribution.scale.to(device)

        sigma, mu = torch.split(self.conv_sigma_mu(x), self._channels, dim=-3)

        self.last_sigma, self.last_mu = sigma, mu

        if self.training:
            z = mu + sigma * self.distribution.sample(mu.shape)
        else:
            z = mu

        return z

    def extra_loss(self) -> Optional[Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, float]]]]:
        if self._kl_loss_weight:
            sigma, mu = self.last_sigma, self.last_mu
            loss_kl = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim=1))
            return {
                "loss_kl": (loss_kl, self._kl_loss_weight)
            }


class ScriptedAE(nn.Module):

    COMMANDS = {
        # add conv layer
        # conv: ch*2, ch/2, ch=32
        "ch": re.compile(r"ch\s*([*/=])\s*(\d+)"),
        # add downsample layer (upsample in decoder)
        # pixel-unshuffle: down2
        "down": re.compile(r"down(\d+)"),
        # add residual downsample layer (from deep-compression-autoencoder)
        # residual pixel-unshuffle: rdown2
        "rdown": re.compile(r"rdown(\d+)"),
        # change kernel size for next convs
        # kernel_size: ks=7
        "ks": re.compile(r"ks=(\d+)"),
        # set number of skip-conv-layers for next convs
        # skip_conv_layers: scl=3
        "scl": re.compile(r"scl=(\d+)"),
        # set polynomial order for next convs (use Conv2dPoly)
        # poly_order: po=3
        "po": re.compile(r"po=(\d+)"),
        # add attention layer with self-invented query/keys
        # attention
        "aqk": re.compile(r"aqk"),
        # add attention layer
        "a": re.compile(r"a"),
        # lightweight multiscale linear attention (from EfficientViT)
        "mla": re.compile(r"mla"),
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
            norm: Optional[str] = None,
            skip_conv_layers: int = 0,
            poly_order: int = 0,
            variational: bool = False,
            kl_loss_weight: float = 1.,
    ):
        assert kernel_size >= 1, f"Got kernel_size={kernel_size}"

        super().__init__()
        self._variational = variational
        self.encoder = nn.Sequential()

        self.decoder = nn.Sequential()
        self.script = "|".join(filter(bool, (l.strip() for l in script.splitlines())))

        ch = channels
        lines = self.script.split("|")
        for line_idx, line in enumerate(lines):
            is_last_encoder_channel = line_idx == len(lines) - 1
            is_last_decoder_channel = len(self.decoder) == 0

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

                pad = padding
                if pad is None:
                    pad = (kernel_size - 1) // 2

                if add_encoder:
                    self.encoder.append(ScriptAELayer(
                        num_in=ch,
                        num_out=new_ch,
                        kernel_size=kernel_size,
                        padding=pad,
                        activation=final_encoder_activation if is_last_encoder_channel else activation,
                        norm=norm if not is_last_encoder_channel else None,
                        poly_order=poly_order,
                        skip_conv_layers=skip_conv_layers,
                        skip_conv_activation=activation,
                    ))
                if add_decoder:
                    self.decoder.insert(0, ScriptAELayer(
                        num_in=new_ch,
                        num_out=ch,
                        kernel_size=kernel_size,
                        padding=pad,
                        activation=final_decoder_activation if is_last_decoder_channel else activation,
                        norm=norm if not is_last_decoder_channel else None,
                        poly_order=poly_order,
                        skip_conv_layers=skip_conv_layers,
                        skip_conv_activation=activation,
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

            elif cmd == "rdown":
                factor = int(args[0])
                if add_encoder:
                    self.encoder.append(DownLayer(ch, ch * factor**2, factor))
                if add_decoder:
                    self.decoder.insert(0, UpLayer(ch * factor**2, ch, factor))
                ch = ch * factor ** 2

            elif cmd == "ks":
                kernel_size = int(args[0])

            elif cmd == "scl":
                skip_conv_layers = int(args[0])

            elif cmd == "po":
                poly_order = int(args[0])

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

            elif cmd == "mla":
                from src.models.efficientvit.ops import LiteMLA, ResidualBlock
                if add_encoder:
                    self.encoder.append(ResidualBlock(
                        main=LiteMLA(ch, ch, norm=(None, "trms2d")),
                        shortcut=nn.Identity(),
                    ))
                if add_decoder:
                    self.decoder.insert(0, ResidualBlock(
                        main=LiteMLA(ch, ch, norm=(None, "trms2d")),
                        shortcut=nn.Identity(),
                    ))

        if variational:
            self.encoder.append(VariationalLayer(ch, kl_loss_weight=kl_loss_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def extra_repr(self) -> str:
        return f"script={repr(self.script)}, variational={self._variational}"

    def extra_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._variational:
            return self.encoder[-1].extra_loss()


if __name__ == "__main__":

    # use it like this:
    ScriptedAE(
        channels=3,
        kernel_size=3,
        script="""
            ch=32
            ch*2
            ch*2
            down2
            ch/4
            down2
            ch/4
            down2
            ch/4
            ch/2
            ch/2
            ch/2
        """
    )
