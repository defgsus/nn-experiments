from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .base_iterable import BaseIterableDataset


class FilterIterableDataset(BaseIterableDataset):
    def __init__(
            self,
            ds: Union[Dataset, IterableDataset],
            *filters: Callable[[Any], bool],
    ):
        self._ds = ds
        self.filters = filters

    def __iter__(self):
        for item in self._ds:
            if self.filters:
                passed = True
                for f in self.filters:
                    if not f(item):
                        passed = False
                        break
                if not passed:
                    continue
            yield item
