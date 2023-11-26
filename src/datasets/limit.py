from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


class LimitDataset(Dataset):

    def __init__(
            self,
            dataset: Dataset,
            size: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return min(self.size, len(self.dataset))

    def __getitem__(self, item):
        if item <= self.size:
            return self.dataset[item]

        raise IndexError(f"{item} is >= {len(self)}")

class LimitIterableDataset(IterableDataset):

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
