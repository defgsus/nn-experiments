from pathlib import Path
from typing import Union, Generator, Optional, Iterable, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset
import torchvision.transforms as VT

from src.util.image import iter_image_patches


class ImagePatchIterableDataset(IterableDataset):
    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset, Iterable[torch.Tensor], Iterable[Tuple[torch.Tensor, ...]]],
            shape: Union[int, Iterable[int]],
            stride: Union[None, int, Iterable[int]] = None,
            padding: Union[int, Iterable[int]] = 0,
            fill: Union[int, float] = 0,
            max_size: Optional[int] = None,
    ):
        """
        Yields patches of each image

        :param dataset: source dataset
        :param shape: one or two ints defining the output shape
        :param stride: one or two ints to define the stride
        :param padding: one or four ints defining the padding
        :param fill: int/float padding value
        :param max_size: optional int to limit the number of yielded patches
        """
        self.dataset = dataset
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        if stride is None:
            self.stride = self.shape
        else:
            self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding if isinstance(padding, int) else tuple(padding)
        self.fill = fill
        self.max_size = max_size

    def __iter__(self):
        count = 0
        for data in self.dataset:
            is_tuple = isinstance(data, (tuple, list))
            if is_tuple:
                image = data[0]
            else:
                image = data

            for patch in iter_image_patches(
                    image=image,
                    shape=self.shape,
                    stride=self.stride,
                    padding=self.padding,
                    fill=self.fill,
            ):
                if is_tuple:
                    yield patch, *data[1:]
                else:
                    yield patch

                count += 1

                if self.max_size is not None and count >= self.max_size:
                    return


def make_image_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path],
        recursive: bool = False,
        stride: Union[None, int, Tuple[int, int]] = None,
        padding: Union[int, Iterable[int]] = 0,
        fill: Union[int, float] = 0,
        max_shuffle: int = 0,
        max_size: Optional[int] = None,
):
    from src.datasets.transform import TransformIterableDataset
    from src.datasets.image_folder import ImageFolderIterableDataset
    from src.datasets.shuffle import IterableShuffle
    from src.util.image import set_image_channels

    transforms = [
        lambda x: x.to(torch.float) / 255. if x.dtype != torch.float else x,
        lambda x: set_image_channels(x, channels=shape[0]),
    ]

    ds_images = TransformIterableDataset(
        ImageFolderIterableDataset(path, recursive=recursive),
        transforms=transforms,
    )
    ds = ImagePatchIterableDataset(
        ds_images,
        shape=shape[-2:],
        stride=shape[-2:] if stride is None else stride,
        max_size=max_size,
        padding=padding,
        fill=fill,
    )
    if max_shuffle:
        ds = IterableShuffle(ds, max_shuffle=max_shuffle)

    return ds
