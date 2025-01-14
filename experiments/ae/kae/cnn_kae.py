import math
from typing import Tuple, Union, Callable

import torch
import torch.nn as nn

from src.models.util import activation_to_module


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


class Reshape(nn.Module):

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.reshape(-1, *self.shape)

    def extra_repr(self):
        return f"shape={self.shape}"


class CNNKAE(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            latent_dim: int,
            channels: Tuple[int, ...] = (24, 32, 48),
            kernel_size: int = 5,
            encoder_kae_order: int = 0,
            decoder_kae_order: int = 0,
            activation: Union[None, str, Callable] = None,
            latent_activation: Union[None, str, Callable] = None,
            output_activation: Union[None, str, Callable] = None,
    ):
        super().__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # --- setup encoder ---

        channels_ = [shape[0], *channels]
        for ch, ch_next in zip(channels_, channels_[1:]):
            self.encoder.append(nn.Conv2d(ch, ch_next, kernel_size))
            if (act := activation_to_module(activation)) is not None:
                self.encoder.append(act)

        # get CNN-stack output size
        with torch.no_grad():
            cnn_out_shape = self.encoder(torch.zeros(1, *shape)).shape[1:]
            cnn_out_dim = math.prod(cnn_out_shape)

        self.encoder.append(nn.Flatten(-3))
        if encoder_kae_order:
            self.encoder.append(KAELayer(cnn_out_dim, latent_dim, encoder_kae_order))
        else:
            self.encoder.append(nn.Linear(cnn_out_dim, latent_dim))

        if (act := activation_to_module(latent_activation)) is not None:
            self.encoder.append(act)

        # --- setup decoder ---

        if decoder_kae_order:
            self.decoder.append(KAELayer(latent_dim, cnn_out_dim, decoder_kae_order))
        else:
            self.decoder.append(nn.Linear(latent_dim, cnn_out_dim))
        self.decoder.append(Reshape(cnn_out_shape))

        channels_ = list(reversed(channels_))
        for i, (ch, ch_next) in enumerate(zip(channels_, channels_[1:])):
            is_last_channel = i == len(channels_) - 2
            self.decoder.append(nn.ConvTranspose2d(ch, ch_next, kernel_size))
            if not is_last_channel:
                if (act := activation_to_module(activation)) is not None:
                    self.decoder.append(act)
            else:
                if (act := activation_to_module(output_activation)) is not None:
                    self.decoder.append(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

