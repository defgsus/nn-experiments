from typing import Union, Generator, Optional, Callable, Any, Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .base_dataset import BaseDataset
from .base_iterable import BaseIterableDataset


class TransformDataset(BaseDataset):
    """
    Transformation on Tensor Dataset

    Optionally convert dtype, multiply by factor and apply list of transforms.
    Can also add features from a DataFrame.

    Transformations only apply to first Tensor in tuple unless `transform_all` is True
    """
    def __init__(
            self,
            source_dataset: Union[Dataset],
            dtype: Optional[torch.dtype] = None,
            multiply: Optional[float] = None,
            transforms: Optional[Iterable[Callable]] = None,
            num_repeat: int = 1,
            features_dataframe: Optional[pd.DataFrame] = None,
            transform_all: bool = False,
    ):
        super().__init__()
        self._source_dataset = source_dataset
        self._dtype = dtype
        self._multiply = multiply
        self._transforms = list(transforms) if transforms is not None else None
        self._num_repeat = num_repeat
        self._features_dataframe = features_dataframe
        self._transform_all = transform_all

    def __len__(self):
        return len(self._source_dataset) * self._num_repeat

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        index = index // self._num_repeat

        item = self._source_dataset[index]

        if isinstance(item, (tuple, int)):
            item, features = item[0], item[1:]

        else:
            item, features = item, None

        item = self._transform(item)
        if self._transform_all and features is not None:
            features = tuple(
                self._transform(f) if isinstance(f, torch.Tensor) else f
                for f in features
            )

        if self._features_dataframe is not None:
            features = torch.Tensor(self._features_dataframe.iloc[index].tolist())
            features = (features, )

        if features is not None:
            return item, *features
        else:
            return item,

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._dtype is not None:
            tensor = tensor.to(self._dtype)

        if self._multiply is not None:
            tensor = tensor * self._multiply

        if self._transforms:
            for t in self._transforms:
                tensor = t(tensor)

        return tensor


class TransformIterableDataset(BaseIterableDataset):
    """
    Transformation on Tensor IterableDataset

    Optionally convert dtype, multiply by factor and apply list of transforms.
    Can also add features from a DataFrame.

    Transformations only apply to first Tensor in tuple
    """
    def __init__(
            self,
            source_dataset: Union[Dataset, IterableDataset],
            dtype: Optional[torch.dtype] = None,
            multiply: Optional[float] = None,
            transforms: Optional[Iterable[Callable]] = None,
            num_repeat: int = 1,
            features_dataframe: Optional[pd.DataFrame] = None,
            remove_tuple: bool = False,
    ):
        super().__init__()
        self._source_dataset = source_dataset
        self._dtype = dtype
        self._multiply = multiply
        self._transforms = list(transforms) if transforms is not None else None
        self._num_repeat = num_repeat
        self._features_dataframe = features_dataframe
        self._remove_tuple = remove_tuple

    def __len__(self):
        return len(self._source_dataset) * self._num_repeat

    def __iter__(self) -> Tuple[torch.Tensor, ...]:
        if self._features_dataframe is not None:
            raise NotImplementedError("__iter__ with features currently not supported")

        for item in self._source_dataset:
            is_tuple = isinstance(item, (tuple, list))
            if is_tuple:
                item, features = item[0], item[1:]
            else:
                item, features = item, None

            for repeat_index in range(self._num_repeat):
                trans_item = self._transform(item)

                if is_tuple and not self._remove_tuple:
                    yield trans_item, *features
                else:
                    yield trans_item

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._dtype is not None:
            tensor = tensor.to(self._dtype)

        if self._multiply is not None:
            tensor = tensor * self._multiply

        if self._transforms:
            for t in self._transforms:
                tensor = t(tensor)

        return tensor
