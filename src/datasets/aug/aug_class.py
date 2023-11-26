import random
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms.v2.functional as VF
import torchvision.transforms.v2 as VT

from src.util import iter_batches


class ImageAugmentClassDataset(Dataset):
    def __init__(
            self,
            dataset: Union[Dataset],
            rotations: Optional[Iterable[float]] = (90., 180., 270.),
            x_shifts: Optional[Iterable[int]] = None,
            y_shifts: Optional[Iterable[int]] = None,
            hflip: bool = False,
            vflip: bool = False,
            invert: bool = False,
            on_value: float = 1.,
            off_value: float = 0.,
    ):
        self.dataset = dataset
        self.augmentations = ["none"]
        if rotations is not None:
            for r in rotations:
                self.augmentations.append(f"rot_{r}")
        if x_shifts is not None:
            for s in x_shifts:
                self.augmentations.append(f"xshift_{s}")
        if y_shifts is not None:
            for s in y_shifts:
                self.augmentations.append(f"yshift_{s}")
        if hflip:
            self.augmentations.append("hflip")
        if vflip:
            self.augmentations.append("vflip")
        if invert:
            self.augmentations.append("invert")
        self.on_value = on_value
        self.off_value = off_value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        item = self.dataset[item]
        if isinstance(item, (tuple, list)):
            image = item[0]
            rest = item[1:]
        else:
            image = item
            rest = tuple()

        augmentation_idx = random.randrange(len(self.augmentations))
        augmentation = self.augmentations[augmentation_idx]

        if augmentation == "none":
            pass

        elif augmentation.startswith("rot_"):
            angle = float(augmentation[4:])
            image = VF.rotate(image, angle)

        elif augmentation.startswith("xshift_"):
            shift = int(augmentation[7:])
            if shift > 0:
                image = VF.pad(image, [shift, 0, 0, 0])[..., :image.shape[-1]]
            elif shift < 0:
                image = VF.pad(image, [0, 0, -shift, 0])[..., -shift:]

        elif augmentation.startswith("yshift_"):
            shift = int(augmentation[7:])
            if shift > 0:
                image = VF.pad(image, [0, shift, 0, 0])[..., :image.shape[-2], :]
            elif shift < 0:
                image = VF.pad(image, [0, 0, 0, -shift])[..., -shift:, :]

        elif augmentation == "vflip":
            image = VF.vflip(image)

        elif augmentation == "hflip":
            image = VF.hflip(image)

        elif augmentation == "invert":
            image = 1.0 - image

        else:
            raise NotImplementedError(f"augmentation '{augmentation}'")

        class_targets = [self.off_value] * len(self.augmentations)
        class_targets[augmentation_idx] = self.on_value
        class_targets = torch.Tensor(class_targets).to(image.device)

        return image, class_targets, *rest
