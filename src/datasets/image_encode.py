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

from src.models.encoder import Encoder2d
from .base_iterable import BaseIterableDataset


class ImageEncodeIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            source_dataset: Union[Dataset, IterableDataset],
            encoder: Encoder2d,
            batch_size: int = 100,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.encoder = encoder
        self.batch_size = batch_size

    def __len__(self):
        return len(self.source_dataset)

    def __iter__(self) -> Generator[Any, None, None]:
        def _encode_batch(items):
            if isinstance(items[0], (list, tuple)):
                images = [i[0] for i in items]
            else:
                images = items

            images = torch.concat([i.unsqueeze(0) for i in images])
            embeddings = self.encoder.encode_image(images)

            for item, embedding in zip(items, embeddings):
                if isinstance(item, (list, tuple)):
                    yield (embedding, item[1:])
                else:
                    yield embedding

        items = []
        for item in self.source_dataset:
            items.append(item)

            if len(items) >= self.batch_size:
                yield from _encode_batch(items)
                items.clear()

        if len(items):
            yield from _encode_batch(items)
