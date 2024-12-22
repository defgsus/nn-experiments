from typing import Tuple, Union, Optional, Iterable, List, Callable, Generator

import torch
from torch.utils.data import Dataset, IterableDataset

from src.datasets.base_dataset import BaseDataset


class FreezeDataset(BaseDataset):

    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
    ):
        self._dataset = dataset
        self._items = None

    def __len__(self):
        self._lazy_freeze()
        return len(self._items)

    def __getitem__(self, i: int):
        self._lazy_freeze()
        return self._items[i]

    def _lazy_freeze(self):
        if self._items is None:
            self._items = list(self._dataset)
