from typing import Tuple, Union, List, Iterable, Optional

import PIL.Image
import PIL.ImageDraw
import torch
from torchvision.utils import make_grid
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


def image_resize_crop(
        image: Union[torch.Tensor, PIL.Image.Image],
        shape: Tuple[int, int],
        interpolation: VF.InterpolationMode = VF.InterpolationMode.NEAREST,
) -> Union[torch.Tensor, PIL.Image.Image]:
    if isinstance(image, PIL.Image.Image):
        width, height = image.width, image.height
    else:
        width, height = image.shape[-1], image.shape[-2]

    if width != shape[1] or height != shape[0]:

        if width < height:
            factor = max(*shape) / width
        else:
            factor = max(*shape) / height

        image = VF.resize(
            image,
            [int(height * factor), int(width * factor)],
            interpolation=interpolation,
        )

        if isinstance(image, PIL.Image.Image):
            width, height = image.width, image.height
        else:
            width, height = image.shape[-1], image.shape[-2]

        if width != shape[1] or height != shape[1]:
            image = VF.center_crop(image, shape)

    return image


def set_image_dtype(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if image.dtype != dtype:
        if image.dtype == torch.uint8:
            image = image.type(dtype) / 255.
        else:
            image = image.type(dtype)

    return image


def set_image_channels(image: torch.Tensor, channels: int) -> torch.Tensor:
    if channels not in (1, 3):
        raise NotImplementedError(f"Conversion to {channels} channels not supported")

    if image.ndim == 2:
        image = image.unsqueeze(0)

    if image.shape[-3] != channels:
        if image.shape[-3] > 3:
            image = image[..., :3, :, :]

        if image.shape[-3] == 1:
            image = image.repeat(*(1 for _ in range(image.ndim - 3)), 3, 1, 1)

        elif image.shape[-3] == 2:
            if channels == 1:
                image = image[..., :1, :, :]
            else:
                image = image[..., :1, :, :].repeat(*(1 for _ in range(image.ndim - 3)), 3, 1, 1)

        elif image.shape[-3] == 3:
            image = VF.rgb_to_grayscale(image, num_output_channels=1)

        else:
            raise NotImplementedError(
                f"Can't convert image.shape={image.shape} to channels={channels}"
            )

    return image


def signed_to_image(data: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if data.ndim == 2:
        data = data.unsqueeze(0)
    elif data.ndim == 3:
        data = data[:1]
    else:
        raise NotImplementedError(f"Can't make signed image of data.shape={data.shape}")

    data_neg = (data < 0).squeeze(0)
    data = data.repeat(3, 1, 1)
    data[1:, data_neg] = 0
    data[0, torch.logical_not(data_neg)] = 0
    image = torch.abs(data)

    if normalize:
        max_value = image.max()
        if max_value:
            image /= max_value
    return image


def get_images_from_iterable(
        images: Iterable[Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]],
        num: int = 8,
        squeezed: bool = False,
) -> List[torch.Tensor]:
    """
    Get at most `num` images from the iterable

    :param images: Can itself be a tensor or DataLoader yielding images or lists/tuples,
        in which case the fist entry is used.

    :return: List[torch.Tensor(1, C, H, W)]
    """
    image_list = []
    for i in images:
        if isinstance(i, (tuple, list)):
            i = i[0]

        if i.ndim == 3:
            image_list.append(i.unsqueeze(0))

        elif i.ndim == 4:
            for idx in range(i.shape[0]):
                image_list.append(i[idx:idx+1])
                if len(image_list) >= num:
                    break
        else:
            raise ValueError(f"Can't handle image with shape {i.shape}")

        if len(image_list) >= num:
            break

    if squeezed:
        for idx, i in enumerate(image_list):
            if i.ndim == 4:
                image_list[idx] = i[0]

    return image_list


def make_grid_labeled(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        labels: Union[bool, Iterable[str]] = True,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: float = 0.0,
        return_pil: bool = False,
        **kwargs,
) -> torch.Tensor:
    grid = make_grid(
        tensor=tensor, nrow=nrow, padding=padding, value_range=value_range,
        scale_each=scale_each, pad_value=pad_value, normalize=normalize,
        **kwargs,
    )

    if labels:
        if isinstance(tensor, (list, tuple)):
            num_images = len(tensor)
            shape = tensor[0].shape
        else:
            assert tensor.ndim == 4, f"make_grid_labeled() only supports [N, C, H, W] shape, got '{tensor.shape}'"
            num_images = tensor.shape[0]
            shape = tensor.shape[1:]

        if labels is True:
            labels = [str(i) for i in range(num_images)]
        else:
            labels = [str(i) for i in labels]

        grid_pil = VF.to_pil_image(grid)
        draw = PIL.ImageDraw.ImageDraw(grid_pil)

        for idx, label in enumerate(labels):
            x = padding + ((idx % nrow) * (shape[-1] + padding))
            y = padding + ((idx // nrow) * (shape[-2] + padding))
            draw.text((x-1, y), label, fill=(0, 0, 0))
            draw.text((x+1, y), label, fill=(0, 0, 0))
            draw.text((x, y-1), label, fill=(0, 0, 0))
            draw.text((x, y+1), label, fill=(0, 0, 0))
            draw.text((x, y), label, fill=(256, 256, 256))

        if return_pil:
            return grid_pil

        grid = VF.to_tensor(grid_pil)

    return grid
