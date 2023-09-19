from typing import Union, Generator, Optional, Tuple

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

    def __iter__(self) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor, ...]], None, None]:
        count = 0
        for data in self.dataset:

            is_tuple = isinstance(data, (tuple, list))
            if is_tuple:
                image = data[0]
            else:
                image = data

            if self.filter(image):
                if self.max_size is None or count < self.max_size:
                    if is_tuple:
                        yield image, *data[1:]
                    else:
                        yield image

                    count += 1

                else:
                    break
