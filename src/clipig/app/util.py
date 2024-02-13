import math
from io import BytesIO
from copy import deepcopy
from typing import List, Generator, Tuple, Optional, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *

import PIL.Image
import torch
import torchvision.transforms.functional as VF


def qimage_to_pil(image: QImage) -> PIL.Image.Image:
    image = image.convertToFormat(QImage.Format_ARGB32)
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    pil_im = PIL.Image.open(BytesIO(buffer.data()))
    return pil_im


def pil_to_qimage(image: PIL.Image.Image) -> QImage:
    buffer = BytesIO()
    image.save(buffer, "PNG")
    buffer.seek(0)
    image = QImage()
    image.loadFromData(buffer.read(), "PNG")
    return image


def qimage_to_torch(image: QImage) -> torch.Tensor:
    """
    Convert QImage to torch tensor

    The image is converted to ARGB32 format before conversion.

    Result is in shape: [C, H, W], where C is (red, green, blue, alpha)

    """
    image = image.convertToFormat(QImage.Format_ARGB32)
    data = image.bits().asarray(image.byteCount())
    data = torch.Tensor(data).reshape(image.height(), image.width(), 4) / 255
    data = data.permute(2, 0, 1)
    return torch.concat([data[2:3, :, :], data[1:2, :, :], data[0:1, :, :], data[3:4, :, :]], dim=0)


def torch_to_qimage(data: torch.Tensor) -> QImage:
    return pil_to_qimage(torch_to_pil(data))


def torch_to_pil(data: torch.Tensor) -> PIL.Image.Image:
    return VF.to_pil_image(data.clamp(0, 1))

    data = data.transpose(1, 2, 0)
    data = data.reshape(data.shape[-1], data.shape[0], data.shape[1])
    if data.shape[0] == 3:
        mode = "RGB"
    elif data.shape[0] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"{data.shape[0]} channels not supported")

    image = PIL.Image.frombytes(mode, (data.shape[-1], data.shape[-2]), data.tobytes())
    return image


def pil_to_torch(image: PIL.Image.Image) -> torch.Tensor:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    return VF.to_tensor(image)


def image_to_qimage(image: Union[PIL.Image.Image, torch.Tensor]) -> QImage:
    if isinstance(image, QImage):
        return image
    elif isinstance(image, PIL.Image.Image):
        return pil_to_qimage(image)
    elif isinstance(image, torch.Tensor):
        return torch_to_qimage(image)
    else:
        raise TypeError(f"Type {type(image).__name__} not supported")
