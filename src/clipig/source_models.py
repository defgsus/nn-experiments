import math
from typing import Tuple, Dict, Optional, Type, List

import torch
import torch.nn as nn
import torchvision.transforms.functional as VF

from src.util.image import *


source_models: Dict[str, Type["SourceModelBase"]] = {}


class SourceModelBase(nn.Module):

    NAME: Optional[str] = None
    PARAMS: List[dict] = []

    def __init_subclass__(cls, **kwargs):
        assert cls.NAME, f"Must specify {cls.__name__}.NAME"
        assert cls.PARAMS, f"Must specify {cls.__name__}.PARAMS"
        if cls.NAME in source_models:
            raise ValueError(
                f"{cls.__name__}.NAME = '{cls.NAME}' is already defined for {source_models[cls.NAME].__name__}"
            )
        source_models[cls.NAME] = cls

    def forward(self):
        raise NotImplementedError

    def randomize(self):
        raise NotImplementedError

    def set_image(self, image: torch.Tensor):
        raise NotImplementedError


def fit_image(image: torch.Tensor, shape: Tuple[int, int, int], dtype: torch.dtype):
    if image.shape[-2:] != torch.Size(shape[-2:]):
        image = VF.resize(image, shape[-2:], VF.InterpolationMode.BICUBIC, antialias=True)
    image = set_image_channels(image, shape[0])
    image = set_image_dtype(image, dtype)
    return image


class PixelModel(SourceModelBase):

    NAME = "pixels"
    PARAMS = [
        *SourceModelBase.PARAMS,
        {
            "name": "channels",
            "type": "select",
            "default": "RGB",
            "choices": ["L", "RGB"]
        },
        {
            "name": "size",
            "type": "int2",
            "default": [224, 224],
            "min": [1, 1],
            "max": [4096, 4096],
        },
    ]

    def __init__(
            self,
            size: Tuple[int, int],
            channels: str,
    ):
        super().__init__()
        channel_map = {"L": 1, "RGB": 3, "HSV": 3}
        num_channels = channel_map.get(channels, 3)
        self.shape = (num_channels, *size)
        self.code = nn.Parameter(torch.empty(self.shape))
        self.randomize()

    def forward(self):
        return self.code.clamp(0, 1)

    @torch.no_grad()
    def randomize(self):
        self.code[:] = torch.randn_like(self.code) * .1 + .3

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        image = fit_image(image, self.shape, self.code.dtype)
        self.code[:] = image


class PixelHSVModel(PixelModel):

    NAME = "pixels_hsv"
    PARAMS = [
        *SourceModelBase.PARAMS,
        {
            "name": "channels",
            "type": "select",
            "default": "HSV",
            "choices": ["L", "HSV"]
        },
        {
            "name": "size",
            "type": "int2",
            "default": [224, 224],
            "min": [1, 1],
            "max": [4096, 4096],
        },
    ]

    def forward(self):
        return hsv_to_rgb(set_image_channels(super().forward(), 3))

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        super().set_image(rgb_to_hsv(set_image_channels(image, 3)))

    @torch.no_grad()
    def randomize(self):
        if self.shape[0] == 3:
            self.code[:1] = torch.rand_like(self.code[:1])
            self.code[1:] = torch.randn_like(self.code[1:]) * .1 + .3
        else:
            self.code[:] = torch.randn_like(self.code) * .1 + .3


class AutoencoderModelHxW(nn.Module):
    def __init__(
            self,
            autoencoder: nn.Module,
            code_size: int,
            shape: Tuple[int, int],
            overlap: Tuple[int, int] = (8, 8),
            std: float = .5,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.code_size = code_size
        self.shape = shape
        self.overlap = tuple(overlap)
        self.std = std
        self.code = nn.Parameter(torch.randn(math.prod(shape), code_size) * std)

    def forward(self):
        images = self.autoencoder.decoder(self.code).clamp(0, 1)

        s = images.shape
        h, w = self.shape[-2:]

        if self.overlap == (0, 0):
            output = torch.zeros_like(images).view(s[-3], s[-2] * h, s[-1] * w)
            for y in range(self.shape[-2]):
                for x in range(self.shape[-1]):
                    output[:, y * s[-2]: (y + 1) * s[-2], x * s[-1]: (x + 1) * s[-1]] = images[y * w + x]

        else:
            output = torch.zeros(
                s[-3],
                s[-2] * h - (h - 1) * self.overlap[-2],
                s[-1] * w - (h - 1) * self.overlap[-1],
            ).to(images.device)
            output_sum = torch.zeros_like(output[0])
            window = get_image_window(s[-2:]).to(output.device)

            for y in range(self.shape[-2]):
                for x in range(self.shape[-1]):
                    yo = y * (s[-2] - self.overlap[-2])
                    xo = x * (s[-1] - self.overlap[-1])
                    output[:, yo: yo + s[-2], xo: xo + s[-1]] = output[:, yo: yo + s[-2], xo: xo + s[-1]] + images[y * w + x] * window
                    output_sum[yo: yo + s[-2], xo: xo + s[-1]] = output_sum[yo: yo + s[-2], xo: xo + s[-1]] + window

            mask = output_sum > 0
            output[:, mask] = output[:, mask] / output_sum[mask]

        return output

    def reset(self):
        with torch.no_grad():
            self.code[:] = torch.randn_like(self.code) * self.std
