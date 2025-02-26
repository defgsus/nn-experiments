import math
from typing import Optional, Tuple, Callable, Union, Dict

import torch
import torch.nn as nn

from src.models.util import activation_to_module, normalization_to_module


class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def extra_repr(self):
        return f"patch_size={self.patch_size}"

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch.shape
        assert W % self.patch_size == 0, f"width must be divisible by patch_size, got {W} / {self.patch_size}"
        assert H % self.patch_size == 0, f"height must be divisible by patch_size, got {H} / {self.patch_size}"

        return (
            batch.permute(0, 2, 3, 1)                     # B, H, W, C
            .unfold(1, self.patch_size, self.patch_size)  # B, H/s, W, C, s
            .unfold(2, self.patch_size, self.patch_size)  # B, H/s, W/s, C, s, s
        )

class Unpatchify(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def extra_repr(self):
        return f"patch_size={self.patch_size}"

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        B, Y, X, C, H, W = batch.shape

        return (
            batch.permute(0, 1, 3, 4, 2, 5)               # B, Y, C, H, X, W
            .reshape(B, Y, C, H, W * X)
            .permute(0, 2, 1, 3, 4)                       # B, C, Y, H, W*X
            .reshape(B, C, H * Y, W * X)
        )


class KAELayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(KAELayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order + 1
        self.addbias = addbias
        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order + 1) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def extra_repr(self):
        return f"input_dim={self.input_dim}, out_dim={self.out_dim}, order={self.order - 1}, addbias={self.addbias}"

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            term = (x_expanded**i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y


class MLPLayer(nn.Module):

    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            kae_order: Optional[int] = None,
            activation: Union[None, str, Callable] = None,
            norm: Union[None, str] = None,
            bias: bool = True,
            residual: bool = True,
    ):
        super().__init__()
        self._residual = residual and channels_in == channels_out

        self.norm = normalization_to_module(norm, channels=channels_in)
        if kae_order is not None:
            self.module = KAELayer(channels_in, channels_out, order=kae_order, addbias=bias)
        else:
            self.module = nn.Linear(channels_in, channels_out, bias=bias)

        self.act = activation_to_module(activation)

    def extra_repr(self):
        return f"residual={self._residual}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.norm is not None:
            y = self.norm(y)
        y = self.module(y)
        if self.act is not None:
            y = self.act(y)
        if self._residual:
            y = y + x
        return y


class MLPMixerLayer(nn.Module):

    def __init__(
            self,
            num_patches: int,
            channels: int,
            type: str = "cnn",  # "cnn", "mlp"
            activation: Union[None, str, Callable] = None,
            bias: bool = True,
            residual: bool = True,
    ):
        super().__init__()
        self._residual = residual
        self.num_patches = num_patches
        self.channels = channels
        self.type = type

        if type == "cnn":
            self.module = nn.Conv1d(num_patches, num_patches, kernel_size=1, bias=bias)
        elif type == "mlp":
            self.module = nn.Linear(num_patches * channels, num_patches * channels, bias=bias)
        else:
            raise ValueError(f"Unknown type '{type}'")

        self.act = activation_to_module(activation)

    def extra_repr(self):
        return f"residual={self._residual}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self.type == "cnn":
            x = x.reshape(x.shape[0] // self.num_patches, self.num_patches, -1)
        else:
            x = x.reshape(x.shape[0] // self.num_patches, -1)
        y = self.module(x)
        if self.act is not None:
            y = self.act(y)
        if self._residual:
            y = y + x
        return y.reshape(shape)


class MixerMLP(nn.Module):

    def __init__(
            self,
            image_shape: Tuple[int, int, int],
            patch_size: int,
            hidden_channels: Tuple[int, ...],
            mixer_at: Tuple[int, ...],
            mixer_type: str = "mlp",
            kae_order_at: Optional[Dict[int, int]] = None,
            activation: Union[None, str, Callable] = None,
            kae_activation: Union[None, str, Callable] = None,
            norm: Union[None, str] = None,
    ):
        image_shape = tuple(image_shape)
        assert image_shape[-1] % patch_size == 0, f"width must be divisible by patch_size, got {image_shape[-1]} / {patch_size}"
        assert image_shape[-2] % patch_size == 0, f"height must be divisible by patch_size, got {image_shape[-2]} / {patch_size}"

        super().__init__()
        self.image_shape = image_shape
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        self.patch_dim = image_shape[0] * (self.patch_size ** 2)
        self.patches_shape = (image_shape[1] // self.patch_size, image_shape[2] // self.patch_size)
        self._last_patch_shape = None
        # self.patcher = nn.Conv2d(in_channels, hidden_channels, kernel_size=patch_size, stride=patch_size)
        self.patchify = Patchify(patch_size)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.unpatchify = Unpatchify(patch_size)

        kae_order_at = kae_order_at or {}
        channels = (self.patch_dim, *self.hidden_channels)
        for i, (ch, next_ch) in enumerate(zip(channels, channels[1:])):
            act = kae_activation if kae_order_at.get(i) else activation
            self.encoder.append(MLPLayer(
                ch, next_ch, activation=act, norm=norm, kae_order=kae_order_at.get(i)
            ))
            self.decoder.insert(0, MLPLayer(
                next_ch, ch, activation=act if (i != 0 or i + 1 in mixer_at) else None, norm=norm, kae_order=kae_order_at.get(i)
            ))
            if i + 1 in mixer_at:
                self.encoder.append(MLPMixerLayer(
                    math.prod(self.patches_shape), next_ch, type=mixer_type,
                ))
                self.decoder.insert(0, MLPMixerLayer(
                    math.prod(self.patches_shape), next_ch, type=mixer_type, activation=activation
                ))

        self.encoder.append(MLPLayer(next_ch * math.prod(self.patches_shape), next_ch))
        self.decoder.insert(0, MLPLayer(next_ch, next_ch * math.prod(self.patches_shape)))

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch.shape
        assert (C, H, W) == self.image_shape, f"Expected image shape {self.image_shape}, got {(C, H, W)}"

        patch_batch = self.patchify(batch)
        patch_shape = patch_batch.shape
        y = patch_batch.reshape(math.prod(patch_shape[:3]), -1)  # B*X*Y, C*S*S

        for i, module in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                y = y.reshape(B, -1)
            y = module(y)
        return y

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        patch_shape = (
            x.shape[0],
            *self.patches_shape,
            self.image_shape[0],
            self.patch_size,
            self.patch_size,
        )

        y = x
        for i, module in enumerate(self.decoder):
            y = module(y)
            if i == 0:
                y = y.reshape(math.prod(patch_shape[:3]), -1)

        y = y.reshape(patch_shape)
        y = self.unpatchify(y)
        return y

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        y = self.encode(batch)
        y = self.decode(y)
        return y
