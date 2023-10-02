import math
import argparse
import random
from pathlib import Path
from functools import partial
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src.datasets import *
from src.util.image import *


def all_patch_dataset(
        shape: Tuple[int, int, int],
        shuffle: int = 200_000,
        file_shuffle: bool = True,
):
    ds = InterleaveIterableDataset(
        (
            kali_patch_dataset(shape, file_shuffle=file_shuffle),
            photo_patch_dataset(shape, "~/Pictures/photos/", file_shuffle=file_shuffle),
            photo_patch_dataset(shape, "~/Pictures/__diverse/", file_shuffle=file_shuffle),
        )
    )
    if shuffle:
        ds = IterableShuffle(ds, max_shuffle=shuffle)

    return ds


# currently about 3.8M patches
def kali_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path] = Path(__file__).resolve().parent.parent / "db/images/kali",
        file_shuffle: bool = True,
):
    return make_image_patch_dataset(
        #verbose_image=True,
        path=path,

        recursive=True,
        shape=shape,
        #max_images=1,
        max_image_bytes=1024 * 1024 * 1024 * 1,
        scales=partial(_scales_from_image_shape, shape, [2., 1., 1./2., 1./5, 1./10, 1./20., 1./30.]),
        stride=5,#_stride,
        interleave_images=20,
        #image_shuffle=5,
        patch_shuffle=10_000,
        file_shuffle=file_shuffle,
    )


def photo_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path] = Path("~/Pictures/photos").expanduser(),
        recursive: bool = True,
        file_shuffle: bool = True,
):
    def _stride(i_shape: Tuple[int, int]):
        # print(shape)
        size = min(i_shape)
        if size <= 512:
            return 5
        return shape[-2:]

    return make_image_patch_dataset(
        path=path, recursive=recursive,
        shape=shape,
        scales=partial(_scales_from_image_shape, shape, [1./12., 1./6, 1./3, 1.]),
        stride=_stride,
        interleave_images=20,
        file_shuffle=file_shuffle,
        transforms=[lambda x: x.clamp(0, 1)],
    )


def _scales_from_image_shape(
        shape: Tuple[int, ...],
        scales: Tuple[float, ...],
        image_shape: Tuple[int, int],
):
    size = min(image_shape)
    shape_size = min(shape[-2:])
    scale_list = []
    for s in scales:
        if s * size >= shape_size and s * size < 10_000:
            scale_list.append(s)
    return scale_list
