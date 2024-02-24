from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms.functional as VF
import torchvision.transforms as VT

from .base_iterable import BaseIterableDataset


class ImageCombinePatchIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            ds: Union[Dataset, IterableDataset],
            shape: Tuple[int, int],
            num_combine: int = 4,
            black_is_alpha: bool = True,
    ):
        self.ds = ds
        self.shape = shape
        self.num_combine = num_combine
        self.black_is_alpha = black_is_alpha
        self.sub_transforms = VT.Compose([
            VT.RandomVerticalFlip(.3),
            VT.RandomHorizontalFlip(.3),
        ])

    def __iter__(self):
        patches = []
        for patch in self.ds:
            if isinstance(patch, (list, tuple)):
                patch = patch[0]
            patches.append(patch)

            if len(patches) >= self.num_combine:
                yield from self._iter_patch_combinations(patches)
                patches.clear()

        if len(patches) > 1:
            yield from self._iter_patch_combinations(patches)

    def _iter_patch_combinations(self, patches):
        C, H, W = patches[0].shape[-3:]

        for background_idx in range(len(patches)):
            mode = VF.InterpolationMode.NEAREST if torch.randint(0, 2, (1,)).item() else VF.InterpolationMode.BICUBIC
            image = VF.resize(patches[background_idx], self.shape, interpolation=mode, antialias=False)
            for idx in torch.randperm(len(patches)):
                if idx != background_idx:
                    x = torch.randint(0, image.shape[-1] - W, (1,)).item()
                    y = torch.randint(0, image.shape[-2] - H, (1,)).item()
                    patch = self.sub_transforms(patches[idx])
                    if self.black_is_alpha:
                        mask = (patch.sum(dim=0) > 0).float().unsqueeze(0).expand(C, -1, -1)
                        patch = patch + (1. - mask) * image[:, y: y+H, x: x+W]
                    image[:, y: y+H, x: x+W] = patch

            yield image
