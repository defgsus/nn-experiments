from pathlib import Path
import glob
import warnings
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision.datasets import ImageFolder as TorchImageFolder, DatasetFolder
from torchvision.datasets.folder import is_image_file
import torchvision.transforms.functional as VF
import torchvision.transforms as VT

from src.util.image import image_resize_crop, set_image_channels, set_image_dtype
from .base_iterable import BaseIterableDataset


class ImageAugmentation(BaseIterableDataset):

    def __init__(
            self,
            source_dataset: IterableDataset,
            augmentations: Iterable[Callable],
            final_shape: Optional[Tuple[int, int]] = None,
            final_channels: Optional[int] = None,
            final_dtype: Optional[torch.dtype] = torch.float32,
    ):
        assert final_channels in (None, 1, 3)

        super().__init__()
        self.source_dataset = source_dataset
        self.augmentations = list(augmentations)
        self.final_shape = final_shape
        self.final_channels = final_channels
        self.final_dtype = final_dtype

    #def __len__(self):
    #    return len(self.source_dataset) * len(self.augmentations)

    def __getitem__(self, index):
        raise NotImplementedError
    #    return self.augment_image(
    #        self.source_dataset[index],
    #        self.augmentations[index % len(self.augmentations)]
    #    )

    def __iter__(self) -> Generator[Union[PIL.Image.Image, torch.Tensor], None, None]:
        # info = get_worker_info()
        for image in self.source_dataset:
            #yield self.augment_image(image, None)

            for aug in self.augmentations:
                yield self.augment_image(image, aug)

    def augment_image(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            augmentation: Optional[Callable] = None,
    ) -> Union[PIL.Image.Image, torch.Tensor]:

        if self.final_channels is not None:
            image = set_image_channels(image, self.final_channels)

        if augmentation is not None:
            image = augmentation(image)

        if self.final_shape is not None:
            image = image_resize_crop(image, self.final_shape)

        if self.final_dtype is not None and image.dtype != self.final_dtype:
            image = set_image_dtype(image, self.final_dtype)

        return image.clamp(0, 1)
