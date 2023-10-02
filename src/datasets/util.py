from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset


def iter_dataset(dataset: Union[Dataset, IterableDataset]) -> Generator:
    try:
        total = len(dataset)
        dataset[0]
    except:
        total = None

    if total:
        for i in range(total):
            yield dataset[i]

    elif hasattr(dataset, "__iter__"):
        yield from dataset

    else:
        TypeError(f"Dataset {dataset} is not iterable")
