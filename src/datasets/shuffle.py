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


class IterableShuffle(IterableDataset):

    def __init__(
            self,
            source_dataset: Dataset,
            shuffle: bool = True,
            max_shuffle: int = 100,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.shuffle = shuffle
        self.max_shuffle = max_shuffle

    def __iter__(self) -> Generator[Any, None, None]:
        items = []
        for item in self.source_dataset:
            items.append(item)

            if len(items) >= self.max_shuffle:
                idx = random.randrange(len(items))
                yield items[idx]
                items.pop(idx)

        while items:
            idx = random.randrange(len(items))
            yield items[idx]
            items.pop(idx)
