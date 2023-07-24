import sys
sys.path.append("..")

import random
import math
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Iterable, Generator

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src.algo import Space2d, kali2d
from src.util import ImageFilter


class Kali2dDataset(Dataset):
    accumulation_choices = ["none", "mean", "max", "min", "submin", "alternate"]

    def __init__(
            self,
            shape: Tuple[int, int, int],
            size: int = 1_000_000,
            seed: int = 23,
            min_scale: float = 0.,
            max_scale: float = 2.,
            min_offset: float = -2.,
            max_offset: float = 2.,
            min_iterations: int = 1,
            max_iterations: int = 37,
            accumulation_modes: Optional[Iterable[str]] = None,
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
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.accumulation_modes = self.accumulation_choices if accumulation_modes is None else list(accumulation_modes)
        self.dtype = dtype
        self.aa = aa

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx) -> torch.Tensor:
        rng = torch.Generator().manual_seed(idx ^ self.seed)

        space = Space2d(
            shape=self.shape,
            offset=torch.rand(self.shape[0], dtype=self.dtype, generator=rng) * (self.min_offset - self.max_offset) + self.min_offset,
            scale=torch.pow(torch.rand(1, dtype=self.dtype, generator=rng)[0], 3.) * (self.max_scale - self.min_scale) + self.min_scale,
            dtype=self.dtype,
        )

        param=torch.rand(self.shape[0], dtype=self.dtype, generator=rng) * 1.2
        accumulate = self.accumulation_modes[torch.randint(0, len(self.accumulation_modes), (1,), generator=rng)[0]]
        iterations=max(self.min_iterations, min(self.max_iterations,
                                                int(torch.randint(self.min_iterations, self.max_iterations, (1,), generator=rng)[0]) + int(1. / space.scale)
                                                ))
        out_weights = (
                torch.rand((self.shape[0], self.shape[0]), dtype=self.dtype, generator=rng) / math.sqrt(self.shape[0])
                + torch.randn((self.shape[0], self.shape[0]), dtype=self.dtype, generator=rng) * .2
        )
        return kali2d(
            space=space,
            param=param,
            iterations=iterations,
            accumulate=accumulate,
            out_weights=out_weights,
            aa=self.aa,
        )


class Kali2dFilteredIterableDataset(IterableDataset):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            size: int = 1_000_000,
            seed: int = 23,
            min_scale: float = 0.,
            max_scale: float = 2.,
            min_offset: float = -2.,
            max_offset: float = 2.,
            min_iterations: int = 1,
            max_iterations: int = 37,
            accumulation_modes: Optional[Iterable[str]] = None,
            dtype: torch.dtype = torch.float,
            aa: int = 0,
            filter: Optional[ImageFilter] = None,
            filter_shape: Tuple[int, int] = (32, 32),
            filter_aa: int = 0,
    ):
        super().__init__()
        kwargs = dict(
            shape=shape,
            size=size,
            seed=seed,
            min_scale=min_scale,
            max_scale=max_scale,
            min_offset=min_offset,
            max_offset=max_offset,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            accumulation_modes=accumulation_modes,
            dtype=dtype,
            aa=aa,
        )
        self.filter = filter
        self.filter_shape = filter_shape
        self.filter_aa = filter_aa

        self.target_dataset = Kali2dDataset(**kwargs)
        self.test_dataset = None
        if self.filter is not None:
            self.test_dataset = Kali2dDataset(**{
                **kwargs,
                "shape": (self.target_dataset.shape[0], *self.filter_shape),
                "aa": self.filter_aa,
            })

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for i in range(len(self.target_dataset)):

            if self.test_dataset is not None:
                image = self.test_dataset[i]
                if not self.filter(image):
                    continue

            yield self.target_dataset[i]
