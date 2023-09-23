
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Dict

import PIL.Image
import PIL.ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


def iter_image_patches(
        image: torch.Tensor,
        shape: Union[int, Iterable[int]],
        stride: Union[None, int, Iterable[int]] = None,
        padding: Union[int, Iterable[int]] = 0,
        fill: Union[int, float] = 0,
        with_pos: bool = False,
        batch_size: Optional[int] = None,
) -> Generator[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    None, None
]:
    """
    Iterate through patches of an image

    :param image: Tensor of shape [C, H, W]
    :param shape: one or two ints defining the output shape
    :param stride: one or two ints to define the stride
    :param padding: one or four ints defining the padding
    :param fill: int/float padding value
    :param with_pos: bool, yield patch positions as well
    :param batch_size: optional int, if defined, N patches will be batched together
    """
    if image.ndim != 3:
        raise ValueError(f"image.ndim != 3 not supported, got {image.ndim}")

    if isinstance(shape, int):
        shape = (shape, shape)
    else:
        shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError(f"shape must have 2 dimensions, got {shape}")
    if any(s < 1 for s in shape):
        raise ValueError(f"shape < 1 not supported, got {shape}")
    if any(s > imgs for s, imgs in zip(shape, image.shape[-2:])):
        raise ValueError(f"shape exceeds image dimensions, got {shape}, image is {image.shape}")

    if stride is None:
        stride = shape
    elif isinstance(stride, int):
        stride = (stride, stride)
    else:
        stride = tuple(stride)
    if len(stride) != 2:
        raise ValueError(f"stride must have 2 dimensions, got {stride}")
    if any(s < 1 for s in stride):
        raise ValueError(f"stride < 1 not supported, got {stride}")

    if padding:
        image = VF.pad(image, padding, fill=fill)

    patch_batch = []
    pos_batch = []
    height, width = image.shape[-2:]
    for y in range(0, height - shape[0] + 1, stride[0]):
        for x in range(0, width - shape[1] + 1, stride[1]):
            patch = image[:, y: y + shape[0], x: x + shape[1]]

            if batch_size is None:
                if with_pos:
                    yield patch, torch.Tensor([y, x]).to(torch.int64)
                else:
                    yield patch

            else:
                patch_batch.append(patch.unsqueeze(0))
                if with_pos:
                    pos_batch.append(torch.Tensor([[y, x]]).to(torch.int64))

                if len(patch_batch) >= batch_size:
                    if with_pos:
                        yield torch.concat(patch_batch), torch.concat(pos_batch)
                    else:
                        yield torch.concat(patch_batch)
                    patch_batch.clear()
                    pos_batch.clear()

    if len(patch_batch):
        if with_pos:
            yield torch.concat(patch_batch), torch.concat(pos_batch)
        else:
            yield torch.concat(patch_batch)

