from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Dict

import PIL.Image
import PIL.ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from tqdm import tqdm

from .image import get_image_window


def iter_image_patches(
        image: torch.Tensor,
        shape: Union[int, Iterable[int]],
        stride: Union[None, int, Iterable[int]] = None,
        padding: Union[int, Iterable[int]] = 0,
        fill: Union[int, float] = 0,
        with_pos: bool = False,
        batch_size: Optional[int] = None,
        verbose: bool = False,
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
    :param verbose: bool, use tqdm for progress bar
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
    with tqdm(disable=not verbose) as progress:
        for y in range(0, height - shape[0] + 1, stride[0]):
            for x in range(0, width - shape[1] + 1, stride[1]):
                progress.update(1)

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


def map_image_patches(
        image: torch.Tensor,
        function: Callable[[torch.Tensor], torch.Tensor],
        patch_size: Tuple[int, int],
        overlap: Union[int, Tuple[int, int]] = 0,
        batch_size: int = 64,
        auto_pad: bool = True,
        window: Union[bool, Callable] = None,
        verbose: bool = False,
) -> torch.Tensor:
    """
    Pass an image patch-wise through `function` with shape [batch_size, C, *patch_size]
    and return processed image.
    """
    if isinstance(overlap, int):
        overlap = [overlap, overlap]
    stride = [patch_size[0] - overlap[0], patch_size[1] - overlap[1]]

    if window is True:
        window = get_image_window(shape=patch_size)
    elif window:
        window = get_image_window(shape=patch_size, window_function=window)
    else:
        window = None

    grid_size = [sh // st for sh, st in zip(image.shape[-2:], stride)]
    recon_size = [gs * st + ov for gs, st, ov in zip(grid_size, stride, overlap)]

    if auto_pad and any(rs < s for rs, s in zip(recon_size, image.shape[-2:])):
        grid_size = [sh // st + 1 for sh, st in zip(image.shape[-2:], stride)]
        recon_size = [gs * st + ov for gs, st, ov in zip(grid_size, stride, overlap)]
        padding = [rs - sh for rs, sh in zip(recon_size, image.shape[-2:])]
        image = VF.pad(image, [0, 0, padding[1], padding[0]])
    else:
        padding = None

    output = torch.zeros_like(image)
    output_sum = torch.zeros_like(image[0])

    for patch_batch, pos_batch in iter_image_patches(
            image=image,
            shape=patch_size,
            stride=stride,
            batch_size=batch_size,
            with_pos=True,
            verbose=verbose,
    ):
        patch_batch = function(patch_batch)
        for patch, pos in zip(patch_batch, pos_batch):
            if window is not None:
                output_sum[pos[-2]: pos[-2] + patch_size[-2], pos[-1]: pos[-1] + patch_size[-1]] += window
                output[:, pos[-2]: pos[-2] + patch_size[-2], pos[-1]: pos[-1] + patch_size[-1]] += patch * window
            else:
                output_sum[pos[-2]: pos[-2] + patch_size[-2], pos[-1]: pos[-1] + patch_size[-1]] += 1
                output[:, pos[-2]: pos[-2] + patch_size[-2], pos[-1]: pos[-1] + patch_size[-1]] += patch

    mask = output_sum > 0
    output[:, mask] /= output_sum[mask].unsqueeze(0)

    if padding:
        output = output[:, :-padding[0], :-padding[1]]

    return output
