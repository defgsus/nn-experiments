from typing import Tuple

import torch.nn as nn


class Reshape(nn.Module):

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.reshape(-1, *self.shape)

    def extra_repr(self):
        return f"shape={self.shape}"
