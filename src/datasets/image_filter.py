from typing import Union, Generator, Optional

import torch
from torch.utils.data import IterableDataset, Dataset

from src.util.image import ImageFilter


class IterableImageFilterDataset(IterableDataset):

    def __init__(
            self,
            dataset: Union[IterableDataset, Dataset],
            filter: ImageFilter,
            max_size: Optional[int] = None,
    ):
        self.dataset = dataset
        self.filter = filter
        self.max_size = max_size

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        count = 0
        for image in self._iter_dataset():

            if self.filter(image):
                if self.max_size is None or count < self.max_size:
                    yield image
                    count += 1

                else:
                    break

    def _iter_dataset(self) -> Generator[torch.Tensor, None, None]:
        if isinstance(self.dataset, Dataset):
            for i in range(len(self.dataset)):
                yield self.dataset[i]
        else:
            yield from self.dataset
