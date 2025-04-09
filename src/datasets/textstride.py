import random
from typing import Optional, Union, Tuple

from torch.utils.data import Dataset, IterableDataset

from src.datasets.base_iterable import BaseIterableDataset


class TextStrideIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
            stride: Union[int, Tuple[int, int]],
            split_character: Optional[str] = None,
            min_length: int = 100,
            seed: Optional[int] = None,
    ):
        self._dataset = dataset
        self._stride = stride
        self._split_character = split_character
        self._min_length = min_length
        if seed is None:
            self._rng = random
        else:
            self._rng = random.Random(seed)

    def __iter__(self):
        for text in self._dataset:
            while len(text) >= self._min_length:
                if not text:
                    break
                yield text

                stride = self._get_stride()
                if self._split_character is None:
                    text = text[stride:]
                else:
                    count = 0
                    for i, ch in enumerate(text):
                        if ch == self._split_character:
                            count += 1
                            if count >= stride:
                                break
                    text = text[i + 1:]

    def _get_stride(self) -> int:
        if isinstance(self._stride, int):
            return self._stride
        else:
            return self._rng.randint(self._stride[0], self._stride[1])
