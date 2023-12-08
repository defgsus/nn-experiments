from typing import Tuple

import torch.nn as nn

from src.functional import soft_histogram


class HistogramLayer(nn.Module):

    def __init__(self, bins: int, min: float, max: float, sigma: float = 100.):
        super().__init__()
        self.bins = bins
        self.min_value = min
        self.max_value = max
        self.sigma = sigma

    def forward(self, x):
        return soft_histogram(x, self.bins, self.min_value, self.max_value, self.sigma)

    def extra_repr(self):
        return f"bins={self.bins}, min={self.min}, max={self.max}, sigma={self.sigma}"
