from typing import Union, Generator, Optional, Iterable, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset

from src.util.image import iter_image_patches


class ImagePatchIterableDataset(IterableDataset):
    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset, Iterable[torch.Tensor], Iterable[Tuple[torch.Tensor, ...]]],
            shape: Union[int, Iterable[int]],
            stride: Union[None, int, Iterable[int]] = None,
            padding: Union[int, Iterable[int]] = 0,
            fill: Union[int, float] = 0,
    ):
        """
        Yields patches of each image

        :param dataset: source dataset
        :param shape: one or two ints defining the output shape
        :param stride: one or two ints to define the stride
        :param padding: one or four ints defining the padding
        :param fill: int/float padding value
        """
        self.dataset = dataset
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        if stride is None:
            self.stride = self.shape
        else:
            self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding if isinstance(padding, int) else tuple(padding)
        self.fill = fill

    def __iter__(self):
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
