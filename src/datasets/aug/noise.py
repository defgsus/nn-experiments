import random
import math
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms.v2.functional as VF
import torchvision.transforms.v2 as VT

from ..base_dataset import BaseDataset


class ImageNoiseDataset(BaseDataset):

    def __init__(
            self,
            image_dataset: Dataset,
            amt_min: float = .01,
            amt_max: float = .15,
            amt_power: float = 1.,
            amounts_per_arg: Iterable[float] = (1,),
            grayscale_prob: float = .0,
            prob: float = 1.,
    ):
        super().__init__()
        self._image_dataset = image_dataset
        self._amt_min = amt_min
        self._amt_max = amt_max
        self._amt_power = amt_power
        self._amounts_per_arg = amounts_per_arg
        self._grayscale_prob = grayscale_prob
        self._prob = prob

    def __len__(self):
        return len(self._image_dataset)

    def __getitem__(self, index):
        item = self._image_dataset[index]

        is_tuple = isinstance(item, (list, tuple))
        if is_tuple:
            image, *rest = item
        else:
            image, rest = item, []

        amt = math.pow(random.uniform(0, 1), self._amt_power)
        amt = self._amt_min + (self._amt_max - self._amt_min) * amt
        is_grayscale = random.uniform(0, 1) < self._grayscale_prob

        if random.uniform(0, 1) >= self._prob:
            noisy_images = [image for _ in self._amounts_per_arg]
        else:
            noisy_images = []
            noisy_image = image
            for sub_amt in self._amounts_per_arg:
                if is_grayscale:
                    noise = torch.randn_like(image[..., :1, :, :]).repeat(
                        *(1 for _ in range(image.ndim - 3)),
                        image.shape[-3], 1, 1
                    )
                else:
                    noise = torch.randn_like(image)

                noisy_image = (noisy_image + sub_amt * amt * noise).clamp(0, 1)
                noisy_images.append(noisy_image)

        if is_tuple:
            return *noisy_images, *rest
        else:
            if len(noisy_images) == 1:
                return noisy_images[0]
            else:
                return *noisy_images,
