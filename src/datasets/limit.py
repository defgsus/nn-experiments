from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


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
