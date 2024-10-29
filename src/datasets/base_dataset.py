from typing import Optional, Tuple, Union, Iterable, Callable

import torch.utils.data.sampler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as VT


class BaseDataset(Dataset):

    def limit(self, size: int, keep_length: bool = False):
        from .limit import LimitDataset
        return LimitDataset(self, size=size, keep_length=keep_length)

    def offset(self, offset: int):
        from .limit import OffsetDataset
        return OffsetDataset(self, offset=offset)

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
    ):
        from .transform import TransformDataset
        return TransformDataset(self, transforms=transforms, dtype=dtype)

    def resize(self, shape: Tuple[int, int], interpolation=VT.InterpolationMode.NEAREST):
        return self.transform([
            VT.Resize(shape, interpolation=interpolation)
        ])


class WrapDataset(BaseDataset):
    def __init__(self, wrapped_dataset: Dataset):
        self._wrapped_dataset = wrapped_dataset

    def __len__(self):
        return len(self._wrapped_dataset)

    def __getitem__(self, item):
        return self._wrapped_dataset[item]
