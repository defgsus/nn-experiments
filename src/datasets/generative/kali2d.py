import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.algo import Space2d, kali_2d
from src.util.image import ImageFilter

"""
nice parameters:

high-tech cave:
increasing the iterations while zooming in will probably look amazing
{'dtype': torch.float32, 'shape': (3, 128, 128), 'aa': 2, 'offset': tensor([1., 1., 1.]), 'scale': tensor(0.0655), 'param': tensor([0.5423, 0.7953, 0.8081]), 'accumulate': 'submin', 'iterations': 19, 'out_weights': tensor([[ 0.5363,  0.7098,  0.0898],
        [ 0.1441,  0.7822,  0.3769],
        [ 0.1168, -0.0425, -0.1233]])}
"""

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
            with_parameters: bool = False,
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
        self.with_parameters = with_parameters

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx) -> torch.Tensor:
        rng = torch.Generator().manual_seed(idx ^ self.seed)

        parameters = dict(
            dtype=self.dtype,
            shape=self.shape,
            aa=self.aa,
            offset=torch.rand(self.shape[0], dtype=self.dtype, generator=rng) * (self.min_offset - self.max_offset) + self.min_offset,
            scale=torch.pow(torch.rand(1, dtype=self.dtype, generator=rng)[0], 3.) * (self.max_scale - self.min_scale) + self.min_scale,
            param=torch.rand(self.shape[0], dtype=self.dtype, generator=rng) * 1.2,
            accumulate=self.accumulation_modes[torch.randint(0, len(self.accumulation_modes), (1,), generator=rng)[0]],
        )
        parameters.update(dict(
            iterations=max(self.min_iterations, min(self.max_iterations,
                int(torch.randint(self.min_iterations, self.max_iterations, (1,), generator=rng)[0]) + int(1. / parameters["scale"])
            )),
            out_weights = (
                torch.rand((self.shape[0], self.shape[0]), dtype=self.dtype, generator=rng) / math.sqrt(self.shape[0])
                + torch.randn((self.shape[0], self.shape[0]), dtype=self.dtype, generator=rng) * .2
            )
        ))

        image = self.render(parameters)
        if self.with_parameters:
            return image, parameters
        else:
            return image

    @classmethod
    def render(cls, parameters: dict) -> torch.Tensor:
        space = Space2d(
            shape=parameters["shape"],
            offset=parameters["offset"],
            scale=parameters["scale"],
            dtype=parameters["dtype"],
        )
        return kali_2d(
            space=space,
            param=parameters["param"],
            iterations=parameters["iterations"],
            accumulate=parameters["accumulate"],
            out_weights=parameters["out_weights"],
            aa=parameters["aa"],
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
            with_parameters: bool = False,
    ):
        super().__init__()
        self.size = size
        kwargs = dict(
            shape=shape,
            size=int(1e10),
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
            with_parameters=with_parameters,
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
                "with_parameters": False,
            })

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        for i in range(len(self.target_dataset)):

            if self.test_dataset is not None:
                image = self.test_dataset[i]
                if not self.filter(image):
                    continue

            yield self.target_dataset[i]

            count += 1
            if count >= self.size:
                break
