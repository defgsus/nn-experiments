import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.algo import Space2d, menger_sponge_2d
from src.util import ImageFilter


class MengerSponge2dDataset(Dataset):
    available_shapes = ("circle", "square", "stripe", "torus")

    def __init__(
            self,
            shape: Tuple[int, int, int],
            size: int = 1000,
            seed: int = 23,
            min_scale: float = .1,
            max_scale: float = 1.,
            min_offset: float = 0.,
            max_offset: float = 0.,
            rotation_steps: int = 4,
            min_radius: float = .1,
            max_radius: float = .25,
            min_iterations: int = 2,
            max_iterations: int = 10,
            min_recursive_scale: float = 1.,
            max_recursive_scale: float = 3.,
            recursive_offset_steps: int = 1,
            recursive_rotation_steps: int = 4,
            shapes: Optional[Iterable[str]] = None,#("circle", "square"),
            dtype: torch.dtype = torch.float,
            aa: int = 0,
    ):
        super().__init__()
        self.shape = shape
        self._size = size
        self.seed = seed
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.rotation_steps = rotation_steps
        self.min_recursive_scale = min_recursive_scale
        self.max_recursive_scale = max_recursive_scale
        self.recursive_offset_steps = recursive_offset_steps
        self.recursive_rotation_steps = recursive_rotation_steps
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.shapes = self.available_shapes if shapes is None else list(shapes)
        self.dtype = dtype
        self.aa = aa

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx) -> torch.Tensor:
        rng = torch.Generator().manual_seed(idx ^ self.seed)
        def _rand(mi: float, ma: float) -> float:
            return torch.rand(1, generator=rng, dtype=self.dtype) * (ma - mi) + mi

        rotation = math.pi * math.floor(_rand(-1., 1.) * self.rotation_steps) / max(1, self.rotation_steps)

        space = Space2d(
            shape=(2, *self.shape[-2:]),
            offset=_rand(self.min_offset, self.max_offset),
            scale=_rand(self.min_scale, self.max_scale),
            rotate_2d=rotation,
            dtype=self.dtype,
        )

        menger_shape = self.shapes[torch.randint(0, len(self.shapes), (1,), generator=rng).item()]
        iterations = max(self.min_iterations, min(self.max_iterations,
                                                  int(.5 + _rand(self.min_iterations, self.max_iterations))
                                                  ))
        radius = _rand(self.min_radius, self.max_radius)
        scale = _rand(self.min_recursive_scale, self.max_recursive_scale)
        rotation = 360. * math.floor(_rand(-1., 1.) * self.recursive_rotation_steps) / max(1, self.recursive_rotation_steps)
        offset = torch.rand(2, dtype=self.dtype, generator=rng)
        offset = torch.floor(offset * self.recursive_offset_steps) / max(1, self.recursive_offset_steps)
        return menger_sponge_2d(
            space=space,
            shape=menger_shape,
            iterations=iterations,
            scale_factor=scale,
            radius=radius,
            rotate_z_deg=rotation,
            offset=offset,
            aa=self.aa,
        )
