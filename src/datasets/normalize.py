from pathlib import Path
import glob
import warnings
import random
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data.dataset import T_co
from torchvision.datasets import ImageFolder as TorchImageFolder, DatasetFolder
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import pil_to_tensor

from src.util import iter_batches


class NormalizeMaxIterableDataset(IterableDataset):

    def __init__(
            self,
            source_dataset: Union[Dataset, IterableDataset],
            num_samples: int = 1000,
            clamp: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.num_samples = num_samples
        self.clamp = clamp

    def __len__(self):
        return len(self.source_dataset)

    def __iter__(self) -> Generator[Any, None, None]:
        for batch in iter_batches(self.source_dataset, batch_size=self.num_samples):
            is_tuple = isinstance(batch, (list, tuple))

            if is_tuple:
                data = batch[0]
            else:
                data = batch

            with torch.no_grad():
                max_v = data.abs().max()
                if max_v:
                    data = data / max_v

                if self.clamp:
                    data = data.clamp(*self.clamp)

            if is_tuple:
                yield from zip(data, *batch[1:])
            else:
                yield from data
