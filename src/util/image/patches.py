import math
import random
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
from src.util import param_make_tuple


def iter_image_patches(
        image: torch.Tensor,
        shape: Union[int, Iterable[int]],
        stride: Union[None, int, Iterable[int]] = None,
        padding: Union[int, Iterable[int]] = 0,
        fill: Union[int, float] = 0,
        random_offset: Union[None, int, Iterable[int]] = None,
        with_pos: bool = False,
        batch_size: Optional[int] = None,
        verbose: bool = False,
) -> Generator[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    None, None
]:
    """
    Iterate through patches of an image

    optionally returns:
        patch-batch, position-batch (where position is [y, x])

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

    shape = param_make_tuple(shape, 2, "shape")
    if any(s < 1 for s in shape):
        raise ValueError(f"shape < 1 not supported, got {shape}")
    if any(s > imgs for s, imgs in zip(shape, image.shape[-2:])):
        raise ValueError(f"shape exceeds image dimensions, got {shape}, image is {image.shape}")

    if stride is None:
        stride = shape
    else:
        stride = param_make_tuple(stride, 2, "stride")
        if any(s < 1 for s in stride):
            raise ValueError(f"stride < 1 not supported, got {stride}")

    if random_offset is not None:
        random_offset = param_make_tuple(random_offset, 2, "random_offset")

    if padding:
        image = VF.pad(image, padding, fill=fill)

    patch_batch = []
    pos_batch = []
    height, width = image.shape[-2:]
    with tqdm(disable=not verbose) as progress:
        for y in range(0, height - shape[0] + 1, stride[0]):
            for x in range(0, width - shape[1] + 1, stride[1]):
                progress.update(1)

                if random_offset is not None:
                    x = min(x + random.randrange(random_offset[1]), width - shape[1])
                    y = min(y + random.randrange(random_offset[0]), height - shape[0])

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


def iter_random_image_patches(
        image: torch.Tensor,
        shape: Union[int, Iterable[int]],
        count: int,
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
    :param with_pos: bool, yield patch positions as well
    :param batch_size: optional int, if defined, N patches will be batched together
    :param verbose: bool, use tqdm for progress bar
    """
    if image.ndim != 3:
        raise ValueError(f"image.ndim != 3 not supported, got {image.ndim}")

    shape = param_make_tuple(shape, 2, "shape")
    if any(s < 1 for s in shape):
        raise ValueError(f"shape < 1 not supported, got {shape}")
    if any(s > imgs for s, imgs in zip(shape, image.shape[-2:])):
        raise ValueError(f"shape exceeds image dimensions, got {shape}, image is {image.shape}")

    patch_batch = []
    pos_batch = []
    height, width = image.shape[-2:]
    for _ in tqdm(range(count), disable=not verbose):
        y = random.randrange(height - shape[-2])
        x = random.randrange(width - shape[-1])

        patch = image[:, y: y + shape[0], x: x + shape[1]]

        if batch_size is None:
            if with_pos:
                yield patch, torch.Tensor([y, x]).to(torch.int64)
            else:
                yield patch

        else:
            patch_batch.append(patch.unsqueeze(0))
            if with_pos:
                pos_batch.append(torch.LongTensor([[y, x]]))

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
        patch_size: Union[int, Tuple[int, int]],
        overlap: Union[int, Tuple[int, int]] = 0,
        cut_away: Union[int, Tuple[int, int]] = 0,
        batch_size: int = 64,
        auto_pad: bool = True,
        padding_mode: str = "zeros",
        window: Union[bool, torch.Tensor, Callable] = False,
        verbose: bool = False,
) -> torch.Tensor:
    """
    Pass an image patch-wise through `function` with shape [batch_size, C, *patch_size]
    and return processed image.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected image.ndim = 3, got {image.shape}")

    patch_size = param_make_tuple(patch_size, 2, "patch_size")
    overlap = param_make_tuple(overlap, 2, "overlap")
    cut_away = param_make_tuple(cut_away, 2, "cut_away")

    for d in (-1, -2):
        if cut_away[d]:
            if cut_away[d] >= patch_size[d] // 2:
                raise ValueError(f"`cut_away` must be smaller than half the patch_size {patch_size}, got {cut_away}")
            if cut_away[d] * 2 + overlap[d] >= patch_size[d]:
                raise ValueError(
                    f"2 * `cut_away` + `overlap` must be smaller than the patch_size {patch_size}"
                    f", got cut_away={cut_away}, overlap={overlap}"
                )

    for i in (-1, -2):
        if overlap[i] >= patch_size[i]:
            raise ValueError(f"`overlap` must be smaller than the patch_size {patch_size}, got {overlap}")

    LEFT, RIGHT, TOP, BOTTOM = range(4)
    padding = [0, 0, 0, 0]

    is_cut_away = bool(any(cut_away))
    if is_cut_away:
        overlap = tuple(o + c * 2 for o, c in zip(overlap, cut_away))
        padding[LEFT] = cut_away[-1]
        padding[TOP] = cut_away[-2]

    stride = [patch_size[0] - overlap[0], patch_size[1] - overlap[1]]

    if isinstance(window, torch.Tensor):
        if window.shape != patch_size:
            raise ValueError(
                f"`window` must match patch_size {patch_size}, got {window.shape}"
            )
    elif window is True:
        window = get_image_window(shape=patch_size)
    elif callable(window):
        window = get_image_window(shape=patch_size, window_function=window)
    else:
        window = None

    if is_cut_away and window is not None:
        window = window[cut_away[-2]: window.shape[-2] - cut_away[-2], cut_away[-1]: window.shape[-1] - cut_away[-1]]

    image_shape = (image.shape[-2] + padding[TOP], image.shape[-1] + padding[LEFT])

    if auto_pad:
        for d, pad_pos in ((-1, RIGHT), (-2, BOTTOM)):
            grid_size = max(1, int(math.ceil(image_shape[d] / stride[d])))
            while True:
                needed_size = grid_size * stride[d] + overlap[d]
                if needed_size >= image_shape[d]:
                    break
                # print(f"  INCREASED dim={d} needed={needed_size} image={image_shape[d]}")
                grid_size += 1

            if needed_size > image_shape[d]:
                padding[pad_pos] = needed_size - image_shape[d]
                # print(f"  needed > image, added padding {padding[pad_pos]}")

    if any(padding):
        image = F.pad(image, padding, mode=padding_mode)

    # print(f"stride={stride} grid={grid_size} image={image.shape[-2:]} pad={padding} overlap={overlap}")

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
            ps = patch_size
            if is_cut_away:
                patch = patch[..., cut_away[-2]: patch.shape[-2] - cut_away[-2], cut_away[-1]: patch.shape[-1] - cut_away[-1]]
                pos = (pos[0] + cut_away[0], pos[1] + cut_away[1])
                ps = patch.shape[-2:]

            add_pixel = 1.
            if window is not None:
                patch *= window
                add_pixel = window

            output_sum[pos[-2]: pos[-2] + ps[-2], pos[-1]: pos[-1] + ps[-1]] += add_pixel
            output[:, pos[-2]: pos[-2] + ps[-2], pos[-1]: pos[-1] + ps[-1]] += patch

    mask = output_sum > 0
    output[:, mask] /= output_sum[mask].unsqueeze(0)

    if any(padding):
        output = output[
            ...,
            padding[TOP]: output.shape[-2] - padding[BOTTOM],
            padding[LEFT]: output.shape[-1] - padding[RIGHT],
        ]

    return output
