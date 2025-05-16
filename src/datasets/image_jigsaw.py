import math
import random
from typing import Union, Tuple, Optional, List

import torch
from torch.utils.data import Dataset, IterableDataset

from src.util.image import image_minimum_size


from .base_iterable import BaseIterableDataset


class ImageJigsawDataset(BaseIterableDataset):
    def __init__(
            self,
            image_dataset: Union[Dataset, IterableDataset],
            puzzle_size: Tuple[int, int],
            tile_size: Tuple[int, int],
            random_spacing: int = 0,
            num_permutation_classes: int = 10,
            num_permutations_per_image: int = 1,
            seed: Optional[int] = None,
            permutation_class_seed: int = 23,
    ):
        super().__init__()
        self._image_dataset = image_dataset
        self._puzzle_size = puzzle_size
        self._tile_size = tile_size
        self._random_spacing = random_spacing
        self._num_permutation_classes = num_permutation_classes
        self._num_permutations_per_image = num_permutations_per_image
        self._permutations: Optional[List] = None
        self._permutation_class_seed = permutation_class_seed
        if seed is None:
            self._rng = random
        else:
            self._rng = random.Random(seed)

    def __iter__(self):
        if self._permutations is None:
            self._permutations = self._create_permutations()
        for image in self._image_dataset:
            if isinstance(image, (list, tuple)):
                image = image[0]
            # display(VF.to_pil_image(image))
            perm_classes = [i % self._num_permutation_classes for i in range(self._num_permutations_per_image)]
            self._rng.shuffle(perm_classes)
            for perm_class in perm_classes:
                perm_class = self._rng.randrange(self._num_permutation_classes)
                crops = self.create_puzzle_crops(image, perm_class)
                yield crops, perm_class

    def _create_permutations(self):
        num_tiles = self._puzzle_size[0] * self._puzzle_size[1]
        if self._num_permutation_classes > math.factorial(num_tiles):
            raise ValueError(f"num_classes ({self._num_permutation_classes}) is too large for {num_tiles} puzzle tiles")
        classes = set()
        rng = random.Random(self._permutation_class_seed)
        while len(classes) < self._num_permutation_classes:
            indices = list(range(num_tiles))
            rng.shuffle(indices)
            classes.add(tuple(indices))
        return sorted(classes)

    def create_puzzle_crops(self, image: torch.Tensor, permutation_class: int, rng: Optional[random.Random] = None):
        if rng is None:
            rng = self._rng

        crop_shape = (
            self._puzzle_size[0] * (self._tile_size[0] + self._random_spacing),
            self._puzzle_size[1] * (self._tile_size[1] + self._random_spacing),
        )
        #image = image_maximum_size(image, max(crop_shape[-1], crop_shape[-2]) + 20)
        image = image_minimum_size(image, crop_shape[-1] + 5, crop_shape[-2] + 5, whole_steps=False)
        xo = rng.randrange(image.shape[-1] - crop_shape[-1])
        yo = rng.randrange(image.shape[-2] - crop_shape[-2])
        image = image[..., yo: yo + crop_shape[-2], xo: xo + crop_shape[-1]]
        #image = image_resize_crop(image, (crop_shape[0] + 20, crop_shape[1] + 20))

        #display(VF.to_pil_image(image))
        crops = []
        for tile_index in self._permutations[permutation_class]:
            x = tile_index % self._puzzle_size[-1]
            y = tile_index // self._puzzle_size[-1]
            x = x * (self._tile_size[-1] + self._random_spacing) + rng.randrange(self._random_spacing + 1)
            y = y * (self._tile_size[-2] + self._random_spacing) + rng.randrange(self._random_spacing + 1)
            crops.append(image[..., y: y + self._tile_size[-2], x: x + self._tile_size[-1]].unsqueeze(0))

        return torch.concat(crops)

