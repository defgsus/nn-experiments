from pathlib import Path
import glob
import warnings
import random
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .util import iter_dataset
from .base_iterable import BaseIterableDataset


class InterleaveIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            datasets: Iterable[Union[Dataset, IterableDataset]],
            shuffle_datasets: bool = False,
            counts: Iterable[int] = None,
    ):
        super().__init__()
        self.datasets = list(datasets)
        self.shuffle_datasets = bool(shuffle_datasets)
        self.counts = None if counts is None else list(counts)
        if self.counts is not None:
            if len(self.counts) != len(self.datasets):
                raise ValueError(
                    f"Expected `counts` to be of length {len(self.datasets)}, got {len(self.counts)}"
                )
            for i, c in enumerate(counts):
                if c < 1:
                    raise ValueError(
                        f"All elements in `counts` must be >= 1, got {c} at index {i}"
                    )

    def __iter__(self) -> Generator[Any, None, None]:
        dataset_iterables = [
            iter_dataset(ds)
            for ds in self.datasets
        ]
        while dataset_iterables:

            dataset_iterables_and_counts = [
                (ds, self.counts[idx]) if self.counts else (ds, 1)
                for idx, ds in enumerate(dataset_iterables)
            ]
            if self.shuffle_datasets:
                random.shuffle(dataset_iterables_and_counts)

            dataset_iterables.clear()
            for ds_iter, count in dataset_iterables_and_counts:
                try:
                    for i in range(count):
                        yield next(ds_iter)
                    dataset_iterables.append(ds_iter)
                except StopIteration:
                    pass
