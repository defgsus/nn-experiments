import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import activation_to_callable
from src.models.decoder import *


class EnsembleDecoder2d(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            activation: Union[None, str, Callable] = "relu",
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = code_size

        self.conv = DecoderConv2d(
            shape=shape, code_size=code_size, activation=activation,
        )

        self.manifold = ImageManifoldDecoder(
            default_shape=shape[-2:], num_input_channels=code_size, num_output_channels=shape[0],
            pos_embedding_freqs=(7, 17, 77,),
        )

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.manifold(x)
        return F.sigmoid(y1) + y2
