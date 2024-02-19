import random
import math
from typing import Tuple, Union

import torch
import torch.nn as nn


class NoiseTransform(nn.Module):

    def __init__(
            self,
            amt_min: float = .01,
            amt_max: float = .15,
            amt_power: float = 2.,
    ):
        super().__init__()
        self.amt_min = amt_min
        self.amt_max = amt_max
        self.amt_power = amt_power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amt = math.pow(random.uniform(0, 1), self.amt_power)
        amt = self.amt_min + (self.amt_max - self.amt_min) * amt

        return x + amt * torch.randn_like(x)


class RandomQuantization(nn.Module):

    def __init__(
            self,
            min_quantization: float = 0.05,
            max_quantization: float = 0.3,
            clamp_output: Union[bool, Tuple[float, float]] = (0., 1.),
    ):
        super().__init__()
        self.min_quantization = min_quantization
        self.max_quantization = max_quantization
        self.clamp_output = clamp_output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.clamp_output is True:
            mi, ma = image.min(), image.max()
        q = max(0.000001, random.uniform(self.min_quantization, self.max_quantization))
        x = (image / q).round() * q

        if self.clamp_output is True:
            x = x.clamp(mi, ma)
        elif self.clamp_output:
            x = x.clamp(*self.clamp_output)

        return x
