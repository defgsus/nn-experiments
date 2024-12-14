from functools import partial
from typing import Optional, Tuple, Union, Iterable, Callable, Any

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as VT


class BaseIterableDataset(IterableDataset):

    def limit(self, size: int):
        from .limit import LimitIterableDataset
        return LimitIterableDataset(self, size)

    def skip(self, count: int):
        from .limit import SkipIterableDataset
        return SkipIterableDataset(self, count)

    def repeat(self, count: int):
        from .limit import RepeatIterableDataset
        return RepeatIterableDataset(self, count)

    def shuffle(self, max_shuffle: int, *, seed: Optional[int] = None):
        from .shuffle import IterableShuffle
        return IterableShuffle(self, max_shuffle=max_shuffle, seed=seed)

    def sample(self, size: int):
        return next(iter(DataLoader(self, batch_size=size)))

    def interleave(
            self,
            *dataset: Union[Dataset, IterableDataset],
            shuffle: bool = False,
            counts: Iterable[int] = None,
    ):
        from .interleave import InterleaveIterableDataset
        return InterleaveIterableDataset(
            datasets=[self, *dataset],
            shuffle_datasets=shuffle,
            counts=counts,
        )

    def scale(
            self,
            scale: Union[float, Iterable[float], Callable[[Tuple[int, int]], Iterable[float]]],
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
            interpolation: VT.InterpolationMode = VT.InterpolationMode.BILINEAR,
            with_scale: bool = False,
    ):
        from .image_scale import ImageScaleIterableDataset

        if isinstance(scale, (int, float)):
            scale = [scale]

        return ImageScaleIterableDataset(
            self,
            scales=scale,
            min_size=min_size,
            max_size=max_size,
            interpolation=interpolation,
            with_scale=with_scale,
        )

    def transform(
            self,
            transforms: Optional[Iterable[Callable]] = None,
            dtype: Optional[torch.dtype] = None,
            transform_all: bool = False
    ):
        from .transform import TransformIterableDataset
        return TransformIterableDataset(self, transforms=transforms, dtype=dtype, transform_all=transform_all)

    def resize(self, shape: Tuple[int, int], interpolation=VT.InterpolationMode.NEAREST):
        return self.transform([
            VT.Resize(shape, interpolation=interpolation)
        ])

    def center_crop(self, shape: Tuple[int, int], all: bool = False):
        from .base_dataset import _center_crop_item
        return self.transform([partial(_center_crop_item, shape=shape, all=all)])

    def filter(self, *filters: Callable[[Any], bool]):
        from .filter import FilterIterableDataset
        return FilterIterableDataset(self, *filters)
