from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets
import torchvision.transforms.functional as VF

from src.datasets import *
from src.util.image import *


def noise_dataset(shape: Tuple[int, ...], size: int, mean: float = .5, std: float = .5, clamp01: bool = True):
    noise = torch.randn(size, *shape) * std + mean
    if clamp01:
        noise = noise.clamp(0, 1)

    return TensorDataset(noise)

