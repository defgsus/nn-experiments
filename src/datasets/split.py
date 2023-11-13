from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class SplitIterableDataset(IterableDataset):
    def __init__(
            self,
            ds: Union[Dataset, IterableDataset],
            ratio: int = 1,
            train: bool = True,
    ):
        """
        Split dataset into two groups and return one.

        :param ds: source dataset
        :param ratio: integer ratio: X for train, 1 for validation
        :param train: bool, switch between the two groups
        """
        self.ds = ds
        self.ratio = ratio
        self.train = train

    def __iter__(self):
        count = 0
        train = True
        for item in self.ds:
            if train == self.train:
                yield item

            count += 1
            if count >= self.ratio:
                count = 0
                train = not train
