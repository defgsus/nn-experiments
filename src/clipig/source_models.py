from typing import Tuple

import torch
import torch.nn as nn


class PixelModel(nn.Module):
    def __init__(self, shape: Tuple[int, int, int] = (3, 224, 224)):
        super().__init__()
        self.shape = shape
        self.code = nn.Parameter(torch.randn(self.shape) * .1 + .3)

    def forward(self):
        return self.code.clamp(0, 1)

    def reset(self):
        with torch.no_grad():
            self.code[:] = torch.randn_like(self.code) * .1 + .3

