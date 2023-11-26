from typing import Union, Generator, Optional, Callable, Any, Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class ClassFeaturesDataset(Dataset):
    """
    Transforms a Dataset with a single label to an n-dim feature vector
    """
    def __init__(
            self,
            source_dataset: Dataset,
            dtype: torch.dtype = torch.float32,
            num_classes: Optional[int] = None,
            off_value: float = 0.,
            on_value: float = 1.,
            tuple_position: int = 0,
            label_to_index: bool = False,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.dtype = dtype
        self.num_classes = num_classes
        self.off_value = off_value
        self.on_value = on_value
        self.tuple_position = tuple_position
        self.label_to_index = label_to_index
        self._label_mapping = {}

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index) -> Tuple[Any, ...]:
        item = self.source_dataset[index]
        is_tuple = isinstance(item, (tuple, list))
        label = self._get_label(item)

        if label not in self._label_mapping:
            if self.num_classes is None:
                self.num_classes = self._count_classes()

            if self.label_to_index:
                idx = int(label)
                if idx < 0 or idx >= self.num_classes:
                    raise ValueError(f"label {label} used as index is out of range, num_classes=={self.num_classes}")
            else:
                idx = len(self._label_mapping)

            label_features = [self.off_value] * self.num_classes
            label_features[idx] = self.on_value

            self._label_mapping[label] = torch.Tensor(label_features).to(self.dtype)

        label_tuple = self._label_mapping[label], torch.tensor(label)

        if not is_tuple:
            return label_tuple

        return (
            *item[:self.tuple_position],
            *label_tuple,
            *item[self.tuple_position + 1:],
        )

    def _count_classes(self) -> int:
        label_set = set()
        for item in self.source_dataset:
            label = self._get_label(item)
            label_set.add(label)
        return len(label_set)

    def _get_label(self, item):
        if isinstance(item, (tuple, int)):
            if self.tuple_position >= len(item):
                raise ValueError(
                    f"tuple_position is {self.tuple_position} but dataset item tuple has length {len(item)}"
                )
            item = item[self.tuple_position]
        else:
            if self.tuple_position:
                raise ValueError(
                    f"tuple_position is {self.tuple_position} but dataset item is type {type(item).__name__}"
                )

        label = item
        if isinstance(label, int):
            pass
        if label.ndim == 0:
            label = label.item()
        elif label.ndim == 1 and label.shape == (1,):
            label = label[0].item()
        else:
            raise ValueError(f"Can't create class features from label with shape {label.shape}")

        return label
