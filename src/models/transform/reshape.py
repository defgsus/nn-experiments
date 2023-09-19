from typing import Tuple

import torch.nn as nn


class Reshape(nn.Module):

    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)