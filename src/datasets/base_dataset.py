from typing import Optional, Tuple, Union, Iterable, Callable

import torch.utils.data.sampler
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as VT


class BaseDataset(Dataset):

    def limit(self, size: int):
        from .limit import LimitDataset
        return LimitDataset(self, size)

    def sample(self, size: int):
        return next(iter(DataLoader(self, batch_size=size)))

    def shuffle(
            self,
            seed: Optional[int] = None,
    ):
        from .shuffle import ShuffleDataset
        return ShuffleDataset(self, seed=seed)
