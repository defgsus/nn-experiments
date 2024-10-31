from typing import Union, Generator, Optional, Callable, Any, Dict, Iterable, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from .base_dataset import BaseDataset
from .base_iterable import BaseIterableDataset
from src.transforms import RandomCropTuple


class RandomCropAllDataset(BaseDataset):
    """
    Applies torchvision's RandomCrop but with the same crop
    for all same-sized images in a list or tuple.
    """
    def __init__(
            self,
            source_dataset: Dataset,
            size: int,
            padding: Optional[int] = None,
            pad_if_needed: bool = False,
            fill: int = 0,
            padding_mode: str = "constant",
    ):
        super().__init__()
        self._source_dataset = source_dataset
        self._cropper = RandomCropTuple(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
        )

    def __len__(self):
        return len(self._source_dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        return self._cropper(self._source_dataset[index])
