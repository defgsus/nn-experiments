from typing import Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.functional import soft_histogram
from src.models.util import get_loss_callable


class HistogramLoss(nn.Module):

    def __init__(
            self,
            bins: int,
            min: float,
            max: float,
            sigma: float = 100.,
            loss: Union[str, Callable] = "l1",
    ):
        super().__init__()
        self.bins = bins
        self.min_value = min
        self.max_value = max
        self.sigma = sigma
        self.loss_function = get_loss_callable(loss)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output_hist = soft_histogram(output, self.bins, self.min_value, self.max_value, self.sigma)
        target_hist = soft_histogram(target, self.bins, self.min_value, self.max_value, self.sigma)

        # output_hist_norm = torch.linalg.norm(output_hist, dim=-1, keepdim=True)
        target_hist_norm = 0.0000001 + torch.linalg.norm(output_hist, dim=-1, keepdim=True)

        output_hist /= target_hist_norm
        target_hist /= target_hist_norm

        return self.loss_function(output_hist, target_hist)

    def extra_repr(self):
        return f"bins={self.bins}, min={self.min}, max={self.max}, sigma={self.sigma}, loss={self.loss_function}"
