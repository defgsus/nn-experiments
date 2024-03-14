import random
import math
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms.v2.functional as VF
import torchvision.transforms.v2 as VT

from ..base_dataset import BaseDataset
from ..base_iterable import BaseIterableDataset


class SuperResolutionDataset(BaseDataset):
    """
    First argument is down-scaled by `factor` and returned as second argument.
    """
    def __init__(
            self,
            image_dataset: Dataset,
            factor: Union[int, float] = 2,
            interpolation: VF.InterpolationMode = VF.InterpolationMode.NEAREST,
            up_interpolation: Optional[VF.InterpolationMode] = None,
            keep_shape: bool = True,
    ):
        super().__init__()
        self._image_dataset = image_dataset
        self._factor = factor
        self._keep_shape = keep_shape
        self._interpolation = interpolation
        self._up_interpolation = up_interpolation or interpolation

    def __len__(self):
        return len(self._image_dataset)

    def __getitem__(self, index):
        item = self._image_dataset[index]

        is_tuple = isinstance(item, (list, tuple))
        if is_tuple:
            image, *rest = item
        else:
            image, rest = item, []

        down_scaled = VF.resize(
            image,
            size=[max(1, int(s / self._factor)) for s in image.shape[-2:]],
            interpolation=self._interpolation,
            antialias=None,
        )
        if self._keep_shape:
            down_scaled = VF.resize(
                down_scaled,
                size=image.shape[-2:],
                interpolation=self._up_interpolation,
                antialias=None,
            )

        return image, down_scaled, *rest


class SuperResolutionIterableDataset(BaseIterableDataset):
    """
    First argument is down-scaled by `factor` and returned as second argument.
    """
    def __init__(
            self,
            image_dataset: Union[Dataset, IterableDataset],
            factor: Union[int, float] = 2,
            interpolation: VF.InterpolationMode = VF.InterpolationMode.NEAREST,
            up_interpolation: Optional[VF.InterpolationMode] = None,
            keep_shape: bool = True,
    ):
        super().__init__()
        self._image_dataset = image_dataset
        self._factor = factor
        self._keep_shape = keep_shape
        self._interpolation = interpolation
        self._up_interpolation = up_interpolation or interpolation

    def __iter__(self):
        for item in self._image_dataset:

            is_tuple = isinstance(item, (list, tuple))
            if is_tuple:
                image, *rest = item
            else:
                image, rest = item, []

            down_scaled = VF.resize(
                image,
                size=[max(1, int(s / self._factor)) for s in image.shape[-2:]],
                interpolation=self._interpolation,
                antialias=None,
            )
            if self._keep_shape:
                down_scaled = VF.resize(
                    down_scaled,
                    size=image.shape[-2:],
                    interpolation=self._up_interpolation,
                    antialias=None,
                )

            yield image, down_scaled, *rest
