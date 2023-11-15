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


class RpgTileIterableDataset(IterableDataset):

    def __init__(
            self,
            shape: Tuple[int, int, int] = (3, 32, 32),
            directory: str = "~/prog/data/game-art/",
            include: Optional[str] = None,
            exclude: Optional[str] = None,
            even_files: Optional[bool] = None,
            interleave: bool = False,
    ):
        self.shape = shape
        self.directory = directory
        self.interleave = interleave
        self.tilesets = [
            dict(name="Castle2.png", shape=(16, 16)),
            dict(name="overworld_tileset_grass.png", shape=(16, 16)),
            dict(name="apocalypse.png", shape=(16, 16)),
            dict(name="PathAndObjects.png", shape=(32, 32)),
            dict(name="mininicular.png", shape=(8, 8)),
            dict(name="items.png", shape=(16, 16)),
            dict(name="roguelikeitems.png", shape=(16, 16), limit_count=181),
            dict(name="tileset_1bit.png", shape=(16, 16)),
            dict(name="MeteorRepository1Icons_fixed.png", shape=(16, 16), offset=(8, 0), stride=(17, 17)),
            dict(name="DENZI_CC0_32x32_tileset.png", shape=(32, 32)),
            dict(name="goodly-2x.png", shape=(32, 32)),
            dict(name="Fruit.png", shape=(16, 16)),
            dict(name="roguelikecreatures.png", shape=(16, 16)),
            dict(name="metroid-like.png", shape=(16, 16), limit=(128, 1000)),
            dict(name="tilesheet_complete.png", shape=(64, 64)),
            dict(name="tiles-map.png", shape=(16, 16)),
            dict(name="base_out_atlas.png", shape=(32, 32)),
            dict(name="build_atlas.png", shape=(32, 32)),
            dict(name="obj_misk_atlas.png", shape=(32, 32)),
            dict(name="Tile-set - Toen's Medieval Strategy (16x16) - v.1.0.png", shape=(16, 16), limit_count=306),
        ]
        if even_files is True:
            self.tilesets = self.tilesets[::2]
        elif even_files is False:
            self.tilesets = self.tilesets[1::2]

        if include is not None:
            self.tilesets = list(filter(
                lambda t: fnmatch.fnmatch(t["name"], include),
                self.tilesets
            ))
        if exclude is not None:
            self.tilesets = list(filter(
                lambda t: not fnmatch.fnmatch(t["name"], exclude),
                self.tilesets
            ))

    def __iter__(self):
        if not self.interleave:
            for params in self.tilesets:
                yield from self._iter_tiles(**params)
        else:
            iterables = [
                self._iter_tiles(**params)
                for params in self.tilesets
            ]
            while iterables:
                next_iterables = []
                for it in iterables:
                    try:
                        yield next(it)
                        next_iterables.append(it)
                    except StopIteration:
                        pass
                iterables = next_iterables

    def _iter_tiles(
            self, name: str,
            shape: Tuple[int, int],
            offset: Tuple[int, int] = None,
            stride: Optional[Tuple[int, int]] = None,
            limit: Optional[Tuple[int, int]] = None,
            limit_count: Optional[int] = None,
            remove_transparent: bool = True,
    ):
        image = VF.to_tensor(PIL.Image.open(
            (Path(self.directory) / name).expanduser()
        ))

        if image.shape[0] != self.shape[0]:
            if image.shape[0] == 4 and remove_transparent:
                image = image[:3] * image[3].unsqueeze(0)
            image = set_image_channels(image[:3], self.shape[0])

        if limit:
            image = image[..., :limit[0], :limit[1]]
        if offset:
            image = image[..., offset[0]:, offset[1]:]

        count = 0
        for patch in iter_image_patches(image, shape, stride=stride):
            if patch.std(1).mean() > 0.:
                #print(patch.std(1).mean())
                patch = VF.resize(patch, self.shape[-2:], VF.InterpolationMode.NEAREST, antialias=False)
                if limit_count is None or count < limit_count:
                    yield patch
                    count += 1
                else:
                    break


def rpg_tile_dataset(
        shape: Tuple[int, int, int],
        validation: Optional[bool] = None,
        limit: Optional[int] = None,
        fake_size: int = 50_000,
        shuffle: bool = False,
        interleave: bool = False,
        random_shift: int = 0,
        random_flip: bool = False,
) -> IterableDataset:

    if validation:
        num_repeat = 1
    elif limit is not None:
        num_repeat = max(1, fake_size // limit)
    else:
        num_repeat = max(1, fake_size // (8000 if validation is None else 4000))

    ds = RpgTileIterableDataset(
        shape,
        interleave=interleave,
    )

    if validation is True:
        ds = SplitIterableDataset(ds, ratio=10, train=False)
    elif validation is False:
        ds = SplitIterableDataset(ds, ratio=10, train=True)

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
        ds = LimitIterableDataset(ds, min(fake_size, limit * num_repeat))

    return ds
    
    