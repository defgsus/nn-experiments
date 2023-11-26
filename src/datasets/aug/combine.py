import random
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms.functional as VF
import torchvision.transforms as VT

from src.util import iter_batches


class CombineImageAugmentIterableDataset(IterableDataset):
    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
            ratio: float = .5,
            crop_ratio: Union[float, Tuple[float, float]] = .5,
            batch_size: int = 128,
            bool_label: bool = False,
    ):
        assert batch_size > 1
        #num_aug = int(batch_size * ratio)
        #if num_aug < 1:
        #    raise ValueError(f"`batch_size` * `ratio` must be >= 1")

        self.dataset = dataset
        self.ratio = ratio
        self.batch_size = batch_size
        self.bool_label = bool_label
        self.crop_ratio = (crop_ratio, crop_ratio) if isinstance(crop_ratio, (float, int)) else tuple(crop_ratio)

    def __iter__(self):
        for batch in iter_batches(self.dataset, self.batch_size):
            is_tuple = isinstance(batch, (list, tuple))
            if is_tuple:
                images = batch[0]
            else:
                images = batch
                batch = (batch,)

            for image_idx, entry in enumerate(zip(images, *batch[1:])):
                if random.random() > self.ratio:
                    yield entry[0], self._label(False), *entry[1:]
                else:
                    image = entry[0].clone()
                    while True:
                        other_idx = random.randrange(images.shape[0])
                        if other_idx != image_idx:
                            break
                    other_image = images[other_idx]

                    crop_size = [
                        random.uniform(*self.crop_ratio)
                        for i in range(2)
                    ]
                    crop_size = [
                        max(1, min(int(c * image.shape[i + 1]), image.shape[i + 1] - 1))
                        for i, c in enumerate(crop_size)
                    ]
                    #print(crop_size)
                    source_pos = [random.randrange(0, s - crop_size[i]) for i, s in enumerate(other_image.shape[-2:])]
                    target_pos = [random.randrange(0, s - crop_size[i]) for i, s in enumerate(other_image.shape[-2:])]

                    image[:, target_pos[0]: target_pos[0] + crop_size[0], target_pos[1]: target_pos[1] + crop_size[1]] = \
                        other_image[:, source_pos[0]: source_pos[0] + crop_size[0], source_pos[1]: source_pos[1] + crop_size[1]]

                    yield image, self._label(True), *entry[1:]

    def _label(self, value: bool):
        if self.bool_label:
            return value
        else:
            return torch.Tensor([0, 1] if value else [1, 0])
