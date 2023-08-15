import random
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Iterable, Generator
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import numpy as np

from src.datasets import *
from src.util.image import *
from src.util import ImageFilter
from src.algo import IFS


class IFSDataset(Dataset):
    def __init__(
            self,
            shape: Tuple[int, int],
            num_classes: int,
            num_parameters: int = 2,
            start_seed: int = 0,
            num_iterations: int = 10_000,
            alpha: float = 0.1,
            patch_size: int = 1,
    ):
        self.shape = shape
        self.num_classes = num_classes
        self.num_parameters = num_parameters
        self.num_iterations = num_iterations
        self.start_seed = start_seed
        self.alpha = alpha
        self.patch_size = patch_size

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, item) -> Tuple[torch.Tensor, int]:
        seed = item + self.start_seed
        ifs = IFS(
            seed=seed,
            num_parameters=self.num_parameters,
        )
        image = torch.Tensor(ifs.render_image(
            shape=self.shape,
            num_iterations=self.num_iterations,
            alpha=self.alpha,
            patch_size=self.patch_size,
        ))

        return image, seed


class IFSClassIterableDataset(IterableDataset):
    def __init__(
            self,
            shape: Tuple[int, int],
            num_classes: int,
            num_instances_per_class: int = 1,
            num_iterations: int = 10_000,
            start_seed: Optional[int] = None,
            variation_seed: Optional[int] = None,
            alpha: float = 0.1,
            patch_size: int = 1,
            parameter_variation: float = 0.03,
            parameter_variation_max: Optional[float] = None,
            alpha_variation: float = 0.05,
            patch_size_variations: Optional[Iterable[int]] = None,
            num_iterations_variation: int = 0,
            image_filter: Optional[ImageFilter] = None,
            filter_num_iterations: Optional[int] = None,
            filter_shape: Optional[Tuple[int, int]] = None,
            filter_alpha: Optional[float] = None,
    ):
        self.shape = shape
        self.num_classes = num_classes
        self.num_instances_per_class = num_instances_per_class
        self.num_iterations = num_iterations
        self.start_seed = start_seed if start_seed is not None else random.randint(0, int(1e10))
        self.alpha = alpha
        self.patch_size = patch_size
        self.parameter_variation = parameter_variation
        self.parameter_variation_max = parameter_variation_max
        self.alpha_variation = alpha_variation
        self.patch_size_variations = list(patch_size_variations) if patch_size_variations is not None else None
        self.num_iterations_variation = num_iterations_variation

        self.image_filter = image_filter
        self.filter_num_iterations = filter_num_iterations
        self.filter_shape = filter_shape
        self.filter_alpha = filter_alpha

        self.rng = np.random.Generator(np.random.MT19937(
            variation_seed if variation_seed is not None else random.randint(0, int(1e10))
        ))
        self.rng.bytes(42)

    def __len__(self) -> int:
        return self.num_classes * self.num_instances_per_class

    def _iter_class_seeds(self) -> Generator[int, None, None]:
        class_index = 0
        class_count = 0
        while class_count < self.num_classes:
            seed = self.start_seed + class_index

            ifs = IFS(seed=seed)
            class_index += 1

            if self.image_filter is not None:

                image = torch.Tensor(ifs.render_image(
                    shape=self.filter_shape or self.shape,
                    num_iterations=self.filter_num_iterations or self.num_iterations,
                    alpha=self.filter_alpha or self.alpha,
                    patch_size=self.patch_size,
                ))
                if not self.image_filter(image):
                    continue

            yield seed
            class_count += 1

    def __iter__(self) -> Generator[Tuple[torch.Tensor, int], None, None]:
        for class_index, seed in enumerate(self._iter_class_seeds()):

            instance_count = 0
            base_mean = None
            while instance_count < self.num_instances_per_class:
                ifs = IFS(seed=seed)

                alpha = self.alpha
                patch_size = self.patch_size
                num_iterations = self.num_iterations

                if instance_count > 0:
                    t = (instance_count + 1) / self.num_instances_per_class

                    amt = self.parameter_variation
                    if self.parameter_variation_max is not None:
                        amt = amt * (1. - t) + t * self.parameter_variation_max

                    ifs.parameters += amt* self.rng.uniform(-1., 1., ifs.parameters.shape)
                    alpha = max(.001, alpha + self.alpha_variation * self.rng.uniform(-1., 1.))
                    if self.patch_size_variations is not None:
                        patch_size = self.patch_size_variations[self.rng.integers(len(self.patch_size_variations))]
                    if self.num_iterations_variation:
                        num_iterations += self.rng.integers(self.num_iterations_variation)

                image = torch.Tensor(ifs.render_image(
                    shape=self.shape,
                    num_iterations=num_iterations,
                    alpha=alpha,
                    patch_size=patch_size,
                ))

                if base_mean is None:
                    base_mean = image.mean()
                else:
                    mean = image.mean()
                    if mean < base_mean / 1.5:
                        continue

                yield image, seed
                instance_count += 1
