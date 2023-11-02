import random
from typing import Tuple, Union

import torch
import torch.nn as nn

from src.algo.wangtile import random_wang_map, render_wang_map


class RandomWangMap(nn.Module):
    """
    Treats input as 4x4 wang tile template and renders a random wang map
    """
    def __init__(
            self,
            map_size: Tuple[int, int],
            overlap: Union[int, Tuple[int, int]] = 0,
            probability: float = 1.,
    ):
        super().__init__()
        self.map_size = map_size
        self.overlap = overlap
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability < 1.:
            if random.random() > self.probability:
                return x

        if x.ndim == 3:
            return render_wang_map(x, random_wang_map(self.map_size), overlap=self.overlap).to(x)

        elif x.ndim == 4:
            return torch.concat([
                render_wang_map(i, random_wang_map(self.map_size), overlap=self.overlap).unsqueeze(0).to(x)
                for i in x
            ])
        else:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got {x.shape}")
