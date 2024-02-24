from typing import Optional

from torch.utils.data import IterableDataset, DataLoader


class BaseIterableDataset(IterableDataset):

    def limit(self, size: int):
        from .limit import LimitIterableDataset
        return LimitIterableDataset(self, size)

    def shuffle(self, max_shuffle: int, seed: Optional[int] = None):
        from .shuffle import IterableShuffle
        return IterableShuffle(self, max_shuffle=max_shuffle, seed=seed)

    def sample(self, size: int):
        return next(iter(DataLoader(self, batch_size=size)))

    # def freeze(self, size: Optional[int]):