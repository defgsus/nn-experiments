import random
from pathlib import Path
from typing import Union, Generator, Optional, Iterable, Tuple, Callable

import PIL.Image
import torch
from torch.utils.data import IterableDataset, Dataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.util.image import iter_image_patches, set_image_channels
from .base_iterable import BaseIterableDataset


class ImagePatchDataset(Dataset):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            filename: Union[str, Path],
    ):
        self.shape = shape
        self.filename = Path(filename).expanduser()
        self._image = None
        self._count = None

    @property
    def image(self):
        if self._image is None:
            self._image = set_image_channels(
                VF.to_tensor(PIL.Image.open(str(self.filename))),
                self.shape[0],
            )
            self._count = (self._image.shape[-1] // self.shape[-1]) * (self._image.shape[-2] // self.shape[-2])
        return self._image

    def __len__(self):
        self.image
        return self._count

    def __getitem__(self, index: int):
        img = self.image
        sy, sx = self.shape[-2:]
        w, h = img.shape[-1] // sx, img.shape[-2] // sy
        x, y = index % w, index // w
        return img[:, y * sy: (y + 1) * sy, x * sx: (x + 1) * sx]


class ImagePatchIterableDataset(BaseIterableDataset):
    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset, Iterable[torch.Tensor], Iterable[Tuple[torch.Tensor, ...]]],
            shape: Union[int, Iterable[int]],
            stride: Union[None, int, Iterable[int], Callable[[Tuple[int, int]], Union[int, Iterable[int]]]] = None,
            padding: Union[int, Iterable[int]] = 0,
            fill: Union[int, float] = 0,
            interleave_images: Optional[int] = None,
            max_size: Optional[int] = None,
            with_pos: bool = False,
    ):
        """
        Yields patches of each source image

        :param dataset: source dataset
        :param shape: one or two ints defining the output shape
        :param stride: one or two ints to define the stride
        :param padding: one or four ints defining the padding
        :param fill: int/float padding value
        :param interleave_images: optional int,
            number of source images to create patches from at the same time
        :param max_size: optional int to limit the number of yielded patches
        :param with_pos: bool, insert the patch rectangle position as second output argument
        """
        self.dataset = dataset
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        if stride is None:
            self.stride = self.shape
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif callable(stride):
            self.stride = stride
        else:
            self.stride = tuple(stride)
        self.padding = padding if isinstance(padding, int) else tuple(padding)
        self.fill = fill
        self.interleave_images = interleave_images
        self.max_size = max_size
        self.with_pos = bool(with_pos)

    def __iter__(self):
        if self.interleave_images:
            yield from self._iter_interleaved()
            return

        count = 0
        for data in self.dataset:
            is_tuple = isinstance(data, (tuple, list))
            if is_tuple:
                image = data[0]
            else:
                image = data

            for patch in self._iter_image_patches(image):
                if self.with_pos:
                    patch, patch_pos = patch
                    if is_tuple:
                        yield patch, patch_pos, *data[1:]
                    else:
                        yield patch, patch_pos
                else:
                    if is_tuple:
                        yield patch, *data[1:]
                    else:
                        yield patch

                count += 1

                if self.max_size is not None and count >= self.max_size:
                    return

    def _iter_interleaved(self):
        patch_iters = []

        def _to_iter(data):
            is_tuple = isinstance(data, (tuple, list))
            if is_tuple:
                image = data[0]
            else:
                image = data

            return {
                "data": data,
                "is_tuple": is_tuple,
                "iter": self._iter_image_patches(image),
                "image": image,
            }

        dataset_iter = iter(self.dataset)
        count = 0
        while True:
            while len(patch_iters) < self.interleave_images:
                try:
                    data = next(dataset_iter)
                except StopIteration:
                    break

                patch_iters.append(_to_iter(data))

            if not patch_iters:
                break

            drop_patch_iters = set()
            for i, patch_iter in enumerate(patch_iters):
                patch = None
                nothing_more = False
                while patch is None and not nothing_more:
                    try:
                        patch = next(patch_iter["iter"])
                    except StopIteration:
                        try:
                            data = next(dataset_iter)
                        except StopIteration:
                            nothing_more = True
                            continue

                        patch_iters[i] = patch_iter = _to_iter(data)

                if patch is None:
                    drop_patch_iters.add(i)
                    continue

                if self.with_pos:
                    patch, patch_pos = patch
                    if patch_iter["is_tuple"]:
                        yield patch, patch_pos, *patch_iter["data"][1:]
                    else:
                        yield patch, patch_pos
                else:
                    if patch_iter["is_tuple"]:
                        yield patch, *patch_iter["data"][1:]
                    else:
                        yield patch

                count += 1

                if self.max_size is not None and count >= self.max_size:
                    return

            if drop_patch_iters:
                patch_iters = [
                    p for i, p in enumerate(patch_iters)
                    if i not in drop_patch_iters
                ]

    def _iter_image_patches(self, image):
        if min(image.shape[-2:]) < min(self.shape):
            return

        if callable(self.stride):
            stride = self.stride(image.shape[-2:])
        else:
            stride = self.stride

        yield from iter_image_patches(
            image=image,
            shape=self.shape,
            stride=stride,
            padding=self.padding,
            fill=self.fill,
            with_pos=self.with_pos,
        )


def make_image_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path, Iterable[Union[str, Path]]],
        recursive: bool = False,
        scales: Union[Iterable[float], Callable[[Tuple[int, int]], Iterable[float]]] = None,
        transforms: Optional[Iterable[Callable]] = None,
        stride: Union[None, int, Tuple[int, int], Callable[[Tuple[int, int]], Union[int, Iterable[int]]]] = None,
        padding: Union[int, Iterable[int]] = 0,
        fill: Union[int, float] = 0,
        interpolation: VT.InterpolationMode = VT.InterpolationMode.BILINEAR,
        interleave_images: Optional[int] = None,
        file_shuffle: bool = False,
        image_shuffle: int = 0,
        patch_shuffle: int = 0,
        max_size: Optional[int] = None,
        max_images: Optional[int] = None,
        max_image_bytes: Optional[int] = None,
        with_pos: bool = False,
        with_scale: bool = False,
        with_filename: bool = False,
        verbose_image: bool = False,
):
    from src.datasets import (
        TransformIterableDataset, ImageFolderIterableDataset, IterableShuffle,
        ImageScaleIterableDataset, InterleaveIterableDataset
    )
    from src.util.image import set_image_channels

    if isinstance(path, (str, Path)):
        ds_images = ImageFolderIterableDataset(
            path,
            recursive=recursive,
            with_filename=with_filename,
            max_images=max_images,
            max_bytes=max_image_bytes,
            verbose=verbose_image,
            shuffle=file_shuffle,
        )
    else:
        ds_images = []
        for path in path:
            ds_images.append(ImageFolderIterableDataset(
                path,
                recursive=recursive,
                with_filename=with_filename,
                max_images=max_images,
                max_bytes=max_image_bytes,
                verbose=verbose_image,
                shuffle=file_shuffle,
            ))
        ds_images = InterleaveIterableDataset(ds_images)

    if scales is not None:
        ds_images = ImageScaleIterableDataset(
            ds_images, scales=scales, with_scale=with_scale,
            interpolation=interpolation,
        )

    if image_shuffle:
        ds_images = IterableShuffle(ds_images, max_shuffle=image_shuffle)

    local_transforms = [
        lambda x: x.to(torch.float) / 255. if x.dtype != torch.float else x,
        lambda x: set_image_channels(x, channels=shape[0]),
    ]
    if transforms:
        local_transforms += list(transforms)
    ds_images = TransformIterableDataset(ds_images, transforms=local_transforms)

    ds = ImagePatchIterableDataset(
        ds_images,
        shape=shape[-2:],
        stride=shape[-2:] if stride is None else stride,
        max_size=max_size,
        padding=padding,
        fill=fill,
        interleave_images=interleave_images,
        with_pos=with_pos,
    )
    if patch_shuffle:
        ds = IterableShuffle(ds, max_shuffle=patch_shuffle)

    return ds


class RandomImagePatchIterableDataset(ImagePatchIterableDataset):
    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset, Iterable[torch.Tensor], Iterable[Tuple[torch.Tensor, ...]]],
            shape: Union[int, Iterable[int]],
            patches_per_image_factor: float = 1.,
            interleave_images: Optional[int] = None,
            with_pos: bool = False,
    ):
        """
        Yields patches of each source image

        :param dataset: source dataset
        :param shape: one or two ints defining the output shape
        :param interleave_images: optional int,
            number of source images to create patches from at the same time
        :param with_pos: bool, insert the patch rectangle position as second output argument
        """
        self.dataset = dataset
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        self.interleave_images = interleave_images
        self.with_pos = bool(with_pos)
        self.patches_per_image_factor = patches_per_image_factor
        self.max_size = None

    def _iter_image_patches(self, image: torch.Tensor):
        if min(image.shape[-2:]) < min(self.shape):
            return

        ps = self.shape
        size = image.shape[-2:]
        count = (size[-2] // ps[-2]) * (size[-1] // ps[-1])
        count = int(count * self.patches_per_image_factor)

        for i in range(count):
            pos = (
                random.randrange(0, max(1, size[-2] - ps[-2])),
                random.randrange(0, max(1, size[-1] - ps[-1]))
            )

            patch = image[:, pos[-2]: pos[-2] + ps[-2], pos[-1]: pos[-1] + ps[-1]]
            if self.with_pos:
                yield patch, pos
            else:
                yield patch

