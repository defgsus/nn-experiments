from functools import partial
from typing import Optional, Tuple, Union, Iterable, Callable, Any

import torch.utils.data.sampler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


class BaseDataset(Dataset):

    def limit(self, size: int, keep_length: bool = False):
        from .limit import LimitDataset
        return LimitDataset(self, size=size, keep_length=keep_length)

    def offset(self, offset: int):
        from .limit import SkipDataset
        return SkipDataset(self, offset=offset)

    def skip(self, offset: int):
        return self.offset(offset)

    def sample(self, size: int):
        return next(iter(DataLoader(self, batch_size=size)))

    def shuffle(self, *, seed: Optional[int] = None):
        from .shuffle import ShuffleDataset
        return ShuffleDataset(self, seed=seed)

    def transform(
            self,
            transforms: Optional[Iterable[Callable]] = None,
            dtype: Optional[torch.dtype] = None,
            num_repeat: int = 1,
            transform_all: bool = False,
    ):
        from .transform import TransformDataset
        return TransformDataset(self, transforms=transforms, dtype=dtype, num_repeat=num_repeat, transform_all=transform_all)

    def resize(self, shape: Tuple[int, int], interpolation=VT.InterpolationMode.NEAREST):
        return self.transform([
            VT.Resize(shape, interpolation=interpolation)
        ])

    def filter(self, *filters: Callable[[Any], bool]):
        from .filter import FilterIterableDataset
        return FilterIterableDataset(self, *filters)

    def center_crop(self, shape: Tuple[int, int], all: bool = False):
        return self.transform([partial(_center_crop_item, shape=shape, all=all)])

    def random_crop_all(self, size: int):
        from .randomcropall import RandomCropAllDataset
        return RandomCropAllDataset(self, size=size)


class WrapDataset(BaseDataset):
    def __init__(self, wrapped_dataset: Dataset):
        self._wrapped_dataset = wrapped_dataset

    def __len__(self):
        return len(self._wrapped_dataset)

    def __getitem__(self, item):
        return self._wrapped_dataset[item]


def _center_crop_item(item: Union[torch.Tensor, Tuple[torch.Tensor, ...]], shape: Tuple[int, int], all: bool):
    if isinstance(item, (tuple, list)):
        if not all:
            image = item[0]
            image = VF.center_crop(image, shape)
            return (image, *item[1:])
        else:
            return tuple(
                VF.center_crop(i, shape) if isinstance(i, torch.Tensor) and i.ndim in (3, 4) else i
                for i in item
            )
    else:
        return VF.center_crop(item, shape)