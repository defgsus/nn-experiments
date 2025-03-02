from pathlib import Path
import glob
import warnings
import random
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data.dataset import T_co
from torchvision.datasets import ImageFolder as TorchImageFolder, DatasetFolder
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import pil_to_tensor

from .base_dataset import BaseDataset
from .base_iterable import BaseIterableDataset


class ShuffleDataset(BaseDataset):

    def __init__(
            self,
            dataset: Dataset,
            max_shuffle: Optional[int] = None,
            seed: Optional[int] = None,
    ):
        """
        1
        2
        3
        4
        5
        6
        7
        8
        9

        :param dataset:
        :param max_shuffle:
        :param seed:
        """
        self._dataset = dataset
        if max_shuffle is None:
            if seed is None:
                generator = None
            else:
                generator = torch.manual_seed(seed)
            self._indices = list(torch.utils.data.sampler.RandomSampler(self, generator=generator))
        else:
            if seed is None:
                rng = random
            else:
                rng = random.Random(seed)
            self._indices = []
            stack = []
            for i in range(len(self)):
                stack.append(i)
                if len(stack) >= max_shuffle:
                    self._indices.append(stack.pop(rng.randrange(len(stack))))
            while stack:
                self._indices.append(stack.pop(rng.randrange(len(stack))))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[self._indices[index]]



class IterableShuffle(BaseIterableDataset):

    def __init__(
            self,
            source_dataset: Union[Dataset, IterableDataset],
            max_shuffle: int = 100,
            seed: Optional[int] = None,
    ):
        super().__init__()
        self._source_dataset = source_dataset
        self._max_shuffle = max_shuffle
        self._rng = random.Random(seed) if seed is not None else random

    def __len__(self):
        return len(self._source_dataset)

    def __iter__(self) -> Generator[Any, None, None]:
        items = []
        for item in self._source_dataset:
            items.append(item)

            if len(items) >= self._max_shuffle:
                idx = self._rng.randrange(len(items))
                yield items[idx]
                items.pop(idx)

        while items:
            idx = self._rng.randrange(len(items))
            yield items[idx]
            items.pop(idx)
