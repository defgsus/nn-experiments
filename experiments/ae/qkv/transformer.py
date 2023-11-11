import math
from collections import OrderedDict
from functools import partial
from typing import Union, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            num_out: int,
            patch_size: int = 8,
            num_layers: int = 4,
            num_heads: int = 4,
            num_hidden: int = 64,
            mlp_dim: Optional[int] = None,
            representation_size: Optional[int] = None,
    ):
        super().__init__()
        assert len(shape) == 3, shape
        assert shape[-2] == shape[-1], shape

        self.shape = shape
        self.transformer = torchvision.models.VisionTransformer(
            image_size=shape[-1],
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=num_hidden,
            num_classes=num_out,
            mlp_dim=mlp_dim or num_hidden,
            representation_size=representation_size,
        )

    def forward(self, x):
        assert x.ndim == 4, x.shape
        B, C, H, W = x.shape
        assert C <= 3, x.shape

        if C < 3:
            x = x.expand(B, 3, H, W)
        return self.transformer(x)
