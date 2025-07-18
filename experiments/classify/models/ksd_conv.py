import math
from typing import List, Union, Optional, Tuple, Generator

import torch
import torch.nn as nn

from src.models.util import activation_to_module


class KSDConvClassificationModel(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: List[int],
            kernel_size: List[int],
            stride: List[int],
            dilation: List[int],
            activation: List[Optional[str]],
            learn_weights_conv: bool,
            learn_weights_linear: bool,
            num_classes: int,
    ):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(channels)):
            conv = nn.Conv2d(
                in_channels=shape[0] if i == 0 else channels[i - 1],
                out_channels=channels[i],
                kernel_size=kernel_size[i],
                stride=stride[i],
                dilation=dilation[i],
                bias=True,
            )
            conv.weight.requires_grad = learn_weights_conv
            self.layers.append(conv)
            if activation[i]:
                self.layers.append(activation_to_module(activation[i]))

        with torch.no_grad():
            in_shape = torch.Size((1, *shape))
            out_shape = self.layers(torch.ones(in_shape)).shape
            flat_size = math.prod(out_shape)
            print(f"{in_shape} ({math.prod(in_shape):,}) -> {out_shape} ({math.prod(out_shape):,}) = {math.prod(out_shape) / math.prod(in_shape)}")

        head = nn.Linear(flat_size, num_classes, bias=True)
        head.weight.requires_grad = learn_weights_linear
        self.layers.append(nn.Flatten(-3))
        self.layers.append(head)

    def forward(self, x):
        return self.layers(x)
