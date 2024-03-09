from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from .base_dataset import BaseDataset
from .base_iterable import BaseIterableDataset


class LimitDataset(BaseDataset):

    def __init__(
            self,
            dataset: Dataset,
            size: int,
            keep_length: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self._size = size
        self._keep_length = keep_length

    def __len__(self):
        if self._keep_length:
            return len(self.dataset)
        return min(self._size, len(self.dataset))

    def __getitem__(self, item):
        if self._keep_length:
            item = item % self._size

        if item <= self._size:
            return self.dataset[item]

        raise IndexError(f"{item} is >= {len(self)}")


class OffsetDataset(BaseDataset):

    def __init__(
            self,
            dataset: Dataset,
            offset: int,
    ):
        super().__init__()
        self.dataset = dataset
        self._offset = offset

    def __len__(self):
        return max(0, len(self.dataset) - self._offset)

    def __getitem__(self, item):
        return self.dataset[item + self._offset]


class LimitIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
            size: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return min(self.size, len(self.dataset))

    def __iter__(self) -> Generator[Any, None, None]:
        count = 0
        for item in self.dataset:
            if count >= self.size:
                break

            yield item
            count += 1
