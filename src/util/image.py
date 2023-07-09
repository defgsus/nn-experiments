from typing import Tuple, Union, List, Iterable

import PIL.Image
import torch
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
    assert channels in (1, 3), f"Got {channels}"

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    if image.shape[-3] != channels:
        if image.shape[-3] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[-3] == 2:
            image = image[0, :, :].repeat(3, 1, 1)
        elif image.shape[-3] == 3:
            image = image.mean(axis=0, keepdim=True)
        elif image.shape[-3] >= 4:
            image = image[..., :3, :, :]
        else:
            raise NotImplementedError(
                f"Can't convert image.shape={image.shape} to channels={channels}"
            )

    return image


def signed_to_image(data: torch.Tensor) -> torch.Tensor:
    if data.ndim == 2:
        data = data.unsqueeze(0)
    elif data.ndim == 3:
        data = data[:1]
    else:
        raise NotImplementedError(f"Can't make image of data.shape={data.shape}")

    data_neg = (data < 0).squeeze(0)
    data = data.repeat(3, 1, 1)
    data[1:, data_neg] = 0
    data[0, torch.logical_not(data_neg)] = 0
    image = torch.abs(data)

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
