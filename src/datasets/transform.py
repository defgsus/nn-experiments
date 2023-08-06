from typing import Union, Generator, Optional, Callable, Any, Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class TransformDataset(Dataset):

    def __init__(
            self,
            source_dataset: Dataset,
            dtype: Optional[torch.dtype] = None,
            multiply: Optional[float] = None,
            transforms: Optional[Iterable[Callable]] = None,
            features_dataframe: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.dtype = dtype
        self.multiply = multiply
        self.transforms = list(transforms) if transforms is not None else None
        self.features_dataframe = features_dataframe

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        item = self.source_dataset[index]

        if isinstance(item, (tuple, int)):
            item, features = item[0], item[1:]

        else:
            item, features = item, None

        item = self._transform(item)

        if self.features_dataframe is not None:
            features = torch.Tensor(self.features_dataframe.iloc[index].tolist())

        if features is not None:
            return item, features
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
