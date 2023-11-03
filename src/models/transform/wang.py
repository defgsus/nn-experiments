import random
from typing import Tuple, Union

import torch
import torch.nn as nn

from src.algo.wangtiles import *


class RandomWangMap(nn.Module):
    """
    Treats input as wang tile template and renders random wang maps
    """
    def __init__(
            self,
            map_size: Tuple[int, int],
            num_colors: int = 2,
            mode: str = "edge",
            overlap: Union[int, Tuple[int, int]] = 0,
            probability: float = 1.,
    ):
        super().__init__()
        self.map_size = map_size
        self.overlap = overlap
        self.probability = probability
        self.wangtiles = WangTiles(get_wang_tile_colors(num_colors), mode=mode)
        self.template = self.wangtiles.create_template((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _render(template):
            if self.probability < 1.:
                if random.random() > self.probability:
                    return template

            self.template.image = template
            return render_wang_map(
                self.template,
                wang_map_stochastic_scanline(self.wangtiles, self.map_size),
                overlap=self.overlap,
            ).to(x)

        if x.ndim == 3:
            return _render(x)

        elif x.ndim == 4:
            return torch.concat([
                _render(i).unsqueeze(0)
                for i in x
            ])
        else:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got {x.shape}")
class RandomWangMap(nn.Module):
    """
    Treats input as wang tile template and renders a random wang map
    """
    def __init__(
            self,
            map_size: Tuple[int, int],
            num_colors: int = 2,
            mode: str = "edge",
            overlap: Union[int, Tuple[int, int]] = 0,
            probability: float = 1.,
    ):
        super().__init__()
        self.map_size = map_size
        self.overlap = overlap
        self.probability = probability
        self.wangtiles = WangTiles(get_wang_tile_colors(num_colors), mode=mode)
        self.template: WangTemplate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        def _render(template):
            if self.probability < 1.:
                if random.random() > self.probability:
                    return template

            if self.template is None or self.template.image.shape != template:
                self.template = self.wangtiles.create_template(template.shape)
            self.template.image = template

            return render_wang_map(
                self.template,
                wang_map_stochastic_scanline(self.wangtiles, self.map_size),
                overlap=self.overlap,
            ).to(x)

        if x.ndim == 3:
            return _render(x)

        elif x.ndim == 4:
            return torch.concat([
                _render(i).unsqueeze(0)
                for i in x
            ])
        else:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got {x.shape}")
