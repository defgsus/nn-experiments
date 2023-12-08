from functools import partial
from typing import Tuple, Optional

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
        normalize_between: Optional[Tuple[float, float]] = None,
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

    if normalize_between is not None:
        transforms.append(partial(globals()["normalize_between"], mi=normalize_between[0], ma=normalize_between[1]))

    return TransformDataset(ds, transforms=transforms)


def normalize_between(x: torch.Tensor, mi: float, ma: float) -> torch.Tensor:
    x_min, x_max = x.min(), x.max()
    if x_min != x_max:
        x = (x - x_min) / (x_max - x_min)

    return x * (ma - mi) + mi



def mnist_dataset(
        train: bool,
        shape: Tuple[int, int, int] = (1, 28, 28),
        interpolation: bool = True,
        normalize_between: Optional[Tuple[float, float]] = None,
) -> Dataset:
    ds = torchvision.datasets.MNIST("~/prog/data/datasets/", train=train)
    return _uint_dataset(
        TensorDataset(ds.data, ds.targets),
        shape=shape,
        interpolation=interpolation,
        normalize_between=normalize_between,
    )


def fmnist_dataset(
        train: bool,
        shape: Tuple[int, int, int] = (1, 28, 28),
        interpolation: bool = True,
        normalize_between: Optional[Tuple[float, float]] = None,
) -> Dataset:
    ds = torchvision.datasets.FashionMNIST("~/prog/data/datasets/", train=train)
    return _uint_dataset(
        TensorDataset(ds.data, ds.targets),
        shape=shape,
        interpolation=interpolation,
        normalize_between=normalize_between,
    )
