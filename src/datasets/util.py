from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset


def iter_dataset(dataset: Union[Dataset, IterableDataset]) -> Generator:
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for i in range(len(dataset)):
            yield dataset[i]

    elif hasattr(dataset, "__iter__"):
        yield from dataset

    else:
        TypeError(f"Dataset {dataset} is not iterable")
