from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets
import torchvision.transforms.functional as VF

from src.datasets import *
from src.util.image import *

DATASETS_ROOT = "~/prog/data/datasets/"


def _uint_dataset(
        ds: Dataset,
        shape: Tuple[int, int, int] = (1, 28, 28),
        default_shape: Tuple[int, int, int] = (1, 28, 28),
        interpolation: bool = True,
) -> Dataset:
    transforms = [
        lambda x: x.unsqueeze(0).float() / 255.,
    ]
    if shape[0] != default_shape[0]:
        transforms.append(lambda x: set_image_channels(x, shape[0]))
    if shape[-2:] != default_shape[-2:]:
        transforms.append(lambda x: VF.resize(
            x, shape[-2:],
            VF.InterpolationMode.BILINEAR if interpolation else VF.InterpolationMode.NEAREST,
            antialias=interpolation
        ))

    return TransformDataset(ds, transforms=transforms)


def mnist_dataset(train: bool, shape: Tuple[int, int, int] = (1, 28, 28), interpolation: bool = True) -> Dataset:
    ds = torchvision.datasets.MNIST("~/prog/data/datasets/", train=train)
    return _uint_dataset(
        TensorDataset(ds.data, ds.targets),
        shape=shape,
        interpolation=interpolation,
    )


def fmnist_dataset(train: bool, shape: Tuple[int, int, int] = (1, 28, 28), interpolation: bool = True) -> Dataset:
    ds = torchvision.datasets.FashionMNIST("~/prog/data/datasets/", train=train)
    return _uint_dataset(
        TensorDataset(ds.data, ds.targets),
        shape=shape,
        interpolation=interpolation,
    )
