from functools import partial
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets
import torchvision.transforms.functional as VF

from src.datasets import *
from src.util.image import *
from src import config

DATASETS_ROOT = config.SMALL_DATASETS_PATH
DO_DOWNLOAD = True


def _dataset(
        ds: Dataset,
        shape: Tuple[int, int, int] = (1, 28, 28),
        default_shape: Tuple[int, int, int] = (1, 28, 28),
        interpolation: bool = True,
        normalize_between: Optional[Tuple[float, float]] = None,
        is_uint: bool = False,
        pre_transforms: List = (),
) -> Dataset:
    transforms = [*pre_transforms]

    if is_uint:
        transforms.append(lambda x: x.unsqueeze(0).float() / 255.)

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
    ds = torchvision.datasets.MNIST(DATASETS_ROOT, train=train, download=DO_DOWNLOAD)
    return _dataset(
        TensorDataset(ds.data, ds.targets),
        shape=shape,
        interpolation=interpolation,
        normalize_between=normalize_between,
        is_uint=True,
    )


def fmnist_dataset(
        train: bool,
        shape: Tuple[int, int, int] = (1, 28, 28),
        interpolation: bool = True,
        normalize_between: Optional[Tuple[float, float]] = None,
) -> Dataset:
    ds = torchvision.datasets.FashionMNIST(DATASETS_ROOT, train=train, download=DO_DOWNLOAD)
    return _dataset(
        TensorDataset(ds.data, ds.targets),
        shape=shape,
        interpolation=interpolation,
        normalize_between=normalize_between,
        is_uint=True,
    )


def cifar10_dataset(
        train: bool,
        shape: Tuple[int, int, int] = (3, 32, 32),
        interpolation: bool = True,
        normalize_between: Optional[Tuple[float, float]] = None,
) -> Dataset:
    ds = torchvision.datasets.CIFAR10(DATASETS_ROOT, train=train, download=DO_DOWNLOAD)
    return _dataset(
        TensorDataset(torch.Tensor(ds.data).permute(0, 3, 1, 2), torch.tensor(ds.targets, dtype=torch.int64)),
        shape=shape,
        interpolation=interpolation,
        normalize_between=normalize_between,
        pre_transforms=[lambda x: x / 255.],
    )


def stl10_dataset(
        train: bool,
        shape: Tuple[int, int, int] = (3, 96, 96),
        interpolation: bool = True,
        normalize_between: Optional[Tuple[float, float]] = None,
) -> Dataset:
    ds = torchvision.datasets.STL10(DATASETS_ROOT, split="train" if train else "test", download=DO_DOWNLOAD)
    return _dataset(
        TensorDataset(torch.Tensor(ds.data) / 255., torch.tensor(ds.labels, dtype=torch.int64)),
        shape=shape,
        interpolation=interpolation,
        normalize_between=normalize_between,
        # pre_transforms=[lambda x: x / 255.],
    )


def flowers102_dataset(
        train: bool,
        shape: Tuple[int, int, int] = (3, 96, 96),
        interpolation: bool = True,
) -> Dataset:
    ds = torchvision.datasets.Flowers102(
        DATASETS_ROOT, split="train" if train else "test", download=DO_DOWNLOAD
    )
    def cropper(item):
        return image_resize_crop(
            item,
            shape=shape[-2:],
            interpolation=VF.InterpolationMode.BILINEAR if interpolation else VF.InterpolationMode.NEAREST,
        )

    return (
        WrapDataset(ds)
        .transform([
            VF.to_tensor,
            cropper,
        ])
    )
    torch.fft.fft2

