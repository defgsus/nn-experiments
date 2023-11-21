import math
import random
import fnmatch
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
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src.datasets import *
from src.util.image import *


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "datasets"


def rpg_tile_dataset_3x32x32(
        shape: Tuple[int, int, int] = (3, 32, 32),
        validation: Optional[bool] = None,
        random_shift: int = 0,
        random_flip: bool = False,
        num_repeat: int = 1,
        interpolation: bool = True,
) -> Dataset:
    filename = DATASET_PATH / f"rpg-3x32x32-uint-{'test' if validation else 'train'}.pt"
    ds = TensorDataset(torch.load(filename))

    transforms = []
    if shape[0] != 3:
        transforms.append(lambda x: set_image_channels(x, shape[0]))
    if shape[-2:] != (32, 32):
        transforms.append(lambda x: VF.resize(
            x, shape[-2:],
            VF.InterpolationMode.BILINEAR if interpolation else VF.InterpolationMode.NEAREST,
            antialias=interpolation
        ))

    if random_shift:
        transforms.extend([
            VT.Pad(random_shift),
            VT.RandomCrop(shape[-2:]),
        ])
    if random_flip:
        transforms.extend([
            VT.RandomHorizontalFlip(.4),
            VT.RandomVerticalFlip(.2),
        ])

    return TransformDataset(
        ds,
        dtype=torch.float32,
        multiply=1./255.,
        transforms=transforms,
        num_repeat=num_repeat,
    )


def rpg_tile_dataset(
        shape: Tuple[int, int, int],
        validation: Optional[bool] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        interleave: bool = False,
        random_shift: int = 0,
        random_flip: bool = False,
        num_repeat: int = 1,
) -> IterableDataset:

    ds = RpgTileIterableDataset(
        shape,
        interleave=interleave,
    )

    if validation is True:
        ds = SplitIterableDataset(ds, ratio=20, train=False)
    elif validation is False:
        ds = SplitIterableDataset(ds, ratio=20, train=True)

    transforms = [
        lambda x: set_image_channels(x, shape[0]),
    ]
    if random_shift:
        transforms.extend([
            VT.Pad(random_shift),
            VT.RandomCrop(shape[-2:]),
        ])
    if random_flip:
        transforms.extend([
            VT.RandomHorizontalFlip(.4),
            VT.RandomVerticalFlip(.2),
        ])

    if shuffle:
        ds = IterableShuffle(ds, 10_000)

    ds = TransformIterableDataset(
        ds,
        transforms=transforms,
        num_repeat=num_repeat,
    )
    if shuffle:
        ds = IterableShuffle(ds, 10_000)

    if limit is not None:
        ds = LimitIterableDataset(ds, limit)

    return ds
    
    