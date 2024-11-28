import math
import random
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
            min: float = 0.,
            max: float = 1.,
            sigma: float = 100.,
            normalize: Union[bool, str] = True,
            loss: Union[str, Callable] = "l1",
    ):
        super().__init__()
        self.bins = bins
        self.min_value = min
        self.max_value = max
        self.sigma = sigma
        self.normalize = normalize
        self.loss_function = get_loss_callable(loss)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert output.ndim == 4, f"Expected 4 dimensions, got {output.shape}"
        assert output.shape == target.shape, f"Shape mismatch: output={output.shape}, target={target.shape}"
        B, C, H, W = output.shape

        output = output.view(B * C, H, W)
        target = target.view(B * C, H, W)
        output_hist = soft_histogram(output, self.bins, self.min_value, self.max_value, self.sigma)
        target_hist = soft_histogram(target, self.bins, self.min_value, self.max_value, self.sigma)
        output_hist = output_hist.view(B, C, self.bins)
        target_hist = target_hist.view(B, C, self.bins)

        if self.normalize:
            if self.normalize is True:
                factor = max(1, math.prod(target.shape[-2:]))
            else:
                factor = 0.0000001 + torch.linalg.norm(output_hist, ord=self.normalize, dim=-1, keepdim=True)

            output_hist = output_hist / factor
            target_hist = target_hist / factor

        return self.loss_function(output_hist, target_hist)

    def extra_repr(self):
        return (
            f"bins={self.bins}, min={self.min}, max={self.max}, sigma={self.sigma}"
            f", normalize={self.normalize} loss={self.loss_function}"
        )
