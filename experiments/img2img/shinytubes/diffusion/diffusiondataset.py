import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.datasets import BaseDataset


class DiffusionDataset(BaseDataset):

    def __init__(
            self,
            ds: Dataset,
            seed: Optional[int] = None,
    ):
        self._ds = ds
        self._diff_amounts = None
        if seed is not None:
            self._diff_amounts = [
                float(i)
                for i in torch.rand(
                    (len(self._ds),),
                    generator=torch.Generator().manual_seed(seed),
                )
            ]

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = self._ds[index]

        if self._diff_amounts is not None:
            amount = self._diff_amounts[index]
        else:
            amount = random.random()

        # amount2 = max(0., amount - .1)

        noise = torch.rand_like(source)

        source = mix(source, noise, amount)
        # target = mix(target, noise, amount)
        return (
            source,
            target,
            torch.tensor(amount, dtype=source.dtype),
        )


def mix(a, b, amount):
    return a * (1. - amount) + b * amount
