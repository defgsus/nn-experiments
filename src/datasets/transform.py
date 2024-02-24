from typing import Union, Generator, Optional, Callable, Any, Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .base_iterable import BaseIterableDataset


class TransformDataset(Dataset):
    """
    Transformation on Tensor Dataset

    Optionally convert dtype, multiply by factor and apply list of transforms.
    Can also add features from a DataFrame.

    Transformations only apply to first Tensor in tuple
    """
    def __init__(
            self,
            source_dataset: Union[Dataset],
            dtype: Optional[torch.dtype] = None,
            multiply: Optional[float] = None,
            transforms: Optional[Iterable[Callable]] = None,
            num_repeat: int = 1,
            features_dataframe: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.dtype = dtype
        self.multiply = multiply
        self.transforms = list(transforms) if transforms is not None else None
        self.num_repeat = num_repeat
        self.features_dataframe = features_dataframe

    def __len__(self):
        return len(self.source_dataset) * self.num_repeat

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        repeat_index = index % self.num_repeat
        index = index // self.num_repeat

        item = self.source_dataset[index]

        if isinstance(item, (tuple, int)):
            item, features = item[0], item[1:]

        else:
            item, features = item, None

        item = self._transform(item)

        if self.features_dataframe is not None:
            features = torch.Tensor(self.features_dataframe.iloc[index].tolist())
            features = (features, )

        if features is not None:
            return item, *features
        else:
            return item,

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)

        if self.multiply is not None:
            tensor = tensor * self.multiply

        if self.transforms:
            for t in self.transforms:
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
        self.source_dataset = source_dataset
        self.dtype = dtype
        self.multiply = multiply
        self.transforms = list(transforms) if transforms is not None else None
        self.num_repeat = num_repeat
        self.features_dataframe = features_dataframe
        self.remove_tuple = remove_tuple

    def __len__(self):
        return len(self.source_dataset) * self.num_repeat

    def __iter__(self) -> Tuple[torch.Tensor, ...]:
        if self.features_dataframe is not None:
            raise NotImplementedError("__iter__ with features currently not supported")

        for item in self.source_dataset:
            is_tuple = isinstance(item, (tuple, list))
            if is_tuple:
                item, features = item[0], item[1:]
            else:
                item, features = item, None

            for repeat_index in range(self.num_repeat):
                trans_item = self._transform(item)

                if is_tuple and not self.remove_tuple:
                    yield trans_item, *features
                else:
                    yield trans_item

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)

        if self.multiply is not None:
            tensor = tensor * self.multiply

        if self.transforms:
            for t in self.transforms:
                tensor = t(tensor)

        return tensor
