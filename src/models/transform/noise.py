import random
import math
from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


class ImageNoise(nn.Module):

    def __init__(
            self,
            amt_min: float = .01,
            amt_max: float = .15,
            amt_power: float = 2.,
            grayscale_prob: float = .1,
            prob: float = 1.,
    ):
        super().__init__()
        self.amt_min = amt_min
        self.amt_max = amt_max
        self.amt_power = amt_power
        self.grayscale_prob = grayscale_prob
        self.prob = prob

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 3:
            return self._noise(input)
        elif input.ndim == 4:
            return torch.concat([
                self._noise(img).unsqueeze(0)
                for img in input
            ])
        else:
            raise ValueError(f"input must have 3 or 4 dimensions, got {input}")

    def _noise(self, image: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) >= self.prob:
            return image

        amt = math.pow(random.uniform(0, 1), self.amt_power)
        amt = self.amt_min + (self.amt_max - self.amt_min) * amt

        if random.uniform(0, 1) < self.grayscale_prob:
            noise = torch.randn_like(image[..., :1, :, :]).repeat(
                *(1 for _ in range(image.ndim - 3)),
                image.shape[-3], 1, 1
            )
        else:
            noise = torch.randn_like(image)

        return (image + amt * noise).clamp(0, 1)


class ImageMultiNoise(nn.Module):

    def __init__(
            self,
            amt_min: float = .01,
            amt_max: float = .15,
            amt_power: float = 2.,
            blur_kernel_size: int = 5,
            blur_sigma_min: float = 0.,
            blur_sigma_max: float = 1.,
            prob: float = 1.,
            channel_modes: Optional[List[str]] = None,
            distribution_modes: Optional[List[str]] = None,
    ):
        super().__init__()
        self.amt_min = amt_min
        self.amt_max = amt_max
        self.amt_power = amt_power
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.blur_kernel_size = blur_kernel_size
        self.prob = prob
        if channel_modes is None:
            channel_modes = ["white", "color"]
        self.channel_modes = channel_modes
        if distribution_modes is None:
            distribution_modes = ["gauss", "positive", "negative", "positive-negative"]
        self.distribution_modes = distribution_modes


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 3:
            return self._noise(input)
        elif input.ndim == 4:
            return torch.concat([
                self._noise(img).unsqueeze(0)
                for img in input
            ])
        else:
            raise ValueError(f"input must have 3 or 4 dimensions, got {input}")

    def _noise(self, image: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) >= self.prob:
            return image

        amt = math.pow(random.uniform(0, 1), self.amt_power)
        amt = self.amt_min + (self.amt_max - self.amt_min) * amt

        channel_mode = random.choice(self.channel_modes)
        distribution_mode = random.choice(self.distribution_modes)

        if distribution_mode == "gauss":
            rand_func = torch.randn_like
        elif distribution_mode in ("positive", "negative"):
            rand_func = torch.rand_like
        elif distribution_mode == "positive-negative":
            rand_func = lambda x: torch.rand_like(x) - torch.rand_like(x)
        else:
            raise ValueError(f"Unknown distribution_mode '{distribution_mode}'")

        if channel_mode == "white":
            noise = rand_func(image[..., :1, :, :]).repeat(
                *(1 for _ in range(image.ndim - 3)),
                image.shape[-3], 1, 1
            )
        elif channel_mode == "color":
            noise = rand_func(image)
        else:
            raise ValueError(f"Unknown channel_mode '{channel_mode}'")

        blur_sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
        if blur_sigma > 0.:
            noise = VF.gaussian_blur(noise, [self.blur_kernel_size, self.blur_kernel_size], [blur_sigma, blur_sigma])

        if distribution_mode == "negative":
            noise = -noise

        return (image + amt * noise).clamp(0, 1)


class RandomQuantization(nn.Module):

    def __init__(
            self,
            min_quantization: float = 0.05,
            max_quantization: float = 0.3,
            clamp_output: Union[bool, Tuple[float, float]] = (0., 1.),
    ):
        super().__init__()
        self.min_quantization = min_quantization
        self.max_quantization = max_quantization
        self.clamp_output = clamp_output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.clamp_output is True:
            mi, ma = image.min(), image.max()
        q = max(0.000001, random.uniform(self.min_quantization, self.max_quantization))
        x = (image / q).round() * q

        if self.clamp_output is True:
            x = x.clamp(mi, ma)
        elif self.clamp_output:
            x = x.clamp(*self.clamp_output)

        return x


class RandomCropHalfImage(nn.Module):

    def __init__(
            self,
            prob: float = 1.,
            null_value: float = 0.,
    ):
        super().__init__()
        self.prob = prob
        self.null_value = null_value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 3:
            return self._crop_half(input)
        elif input.ndim == 4:
            return torch.concat([
                self._crop_half(img).unsqueeze(0)
                for img in input
            ])
        else:
            raise ValueError(f"input must have 3 or 4 dimensions, got {input}")

    def _crop_half(self, image: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) >= self.prob:
            return image

        lrtb = random.randrange(4)
        if lrtb == 0:
            slices = slice(None, None), slice(None, image.shape[-1] // 2)
        elif lrtb == 1:
            slices = slice(None, None), slice(image.shape[-1] // 2, None)
        elif lrtb == 2:
            slices = slice(None, image.shape[-2] // 2), slice(None, None)
        else:
            slices = slice(image.shape[-2] // 2, None), slice(None, None)

        new_image = image + 0
        new_image[:, slices[0], slices[1]] = self.null_value
        return new_image
