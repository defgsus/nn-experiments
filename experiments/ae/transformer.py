import math
import random
from collections import OrderedDict
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import torch.utils.data
import torchvision.models
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            patch_size: Union[int, Tuple[int, int]] = 8,
            stride: Union[None, int, Tuple] = None,
            num_layers: int = 4,
            num_hidden: int = 64,
            num_heads: int = 4,
            mlp_dim: Optional[int] = None,
            dropout: float = 0.1,
            activation: Union[None, str, Callable] = F.relu,
    ):
        super().__init__()
        conv_in = nn.Conv2d(shape[0], num_hidden, kernel_size=patch_size, stride=stride or patch_size)
        conv_out = nn.ConvTranspose2d(num_hidden, shape[0], kernel_size=patch_size, stride=stride or patch_size)
        self.conv_shape = conv_in(torch.empty(1, *shape)).shape[-3:]

        self.proj = nn.Linear(code_size, num_hidden)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=num_hidden,
                nhead=num_heads,
                dim_feedforward=mlp_dim or num_hidden,
                dropout=dropout,
                activation=activation,
            ),
            num_layers=num_layers,
        )
        self.proj_out = nn.Linear(num_hidden, math.prod(self.conv_shape))
        self.patches = conv_out

    def forward(self, x):
        assert x.ndim == 2, x.shape
        y = self.proj(x)
        y = self.transformer(y, y)
        y = self.proj_out(y).view(-1, *self.conv_shape)
        y = self.patches(y)
        return y


class TransformerAutoencoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            patch_size: Union[int, Tuple[int, int]] = 8,
            stride: Union[None, int, Tuple] = None,
            num_layers: int = 4,
            num_hidden: int = 64,
            num_heads: int = 4,
            mlp_dim: Optional[int] = None,
            dropout: float = 0.1,
            activation: Union[None, str, Callable] = F.relu,
    ):
        super().__init__()
        assert len(shape) == 3, shape

        self.shape = shape

        conv_in = nn.Conv2d(shape[0], num_hidden, kernel_size=patch_size, stride=stride or patch_size)
        conv_shape = conv_in(torch.empty(1, *shape)).shape[-3:]

        self.encoder = nn.Sequential(OrderedDict([
            ("patches", conv_in),
            ('flatten', nn.Flatten(-3)),
            ("proj", nn.Linear(math.prod(conv_shape), num_hidden)),
            ("transformer", nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=num_hidden,
                    nhead=num_heads,
                    dim_feedforward=mlp_dim or num_hidden,
                    dropout=dropout,
                    activation=activation,
                ),
                num_layers=num_layers,
            )),
            ("proj_out", nn.Linear(num_hidden, code_size)),
        ]))
        self.decoder = TransformerDecoder(
            shape=shape,
            code_size=code_size,
            patch_size=patch_size,
            stride=stride,
            num_layers=num_layers,
            num_hidden=num_hidden,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        assert x.ndim == 4, x.shape

        y = self.encoder(x)
        return self.decoder(y)
        torchvision.models.ResNet