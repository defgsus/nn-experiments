import math
from io import BytesIO
from typing import Iterable, Callable, Optional

import torch
import torchvision.transforms.functional as VF


class ImageFilter:

    def __init__(
            self,
            min_mean: float = 0.,
            max_mean: float = 0.,
            min_std: float = 0.,
            max_std: float = 0.,
            min_compression_ratio: float = 0.,
            max_compression_ratio: float = 0.,
            min_scaled_compression_ratio: float = 0.,
            max_scaled_compression_ratio: float = 0.,
            scaled_compression_shape: Iterable[int] = (16, 16),
            min_blurred_compression_ratio: float = 0.,
            max_blurred_compression_ratio: float = 0.,
            blurred_compression_kernel_size: Iterable[int] = (11, 11),
            blurred_compression_sigma: float = 10.,
            compression_format: str = "png",
            callable: Optional[Callable[[torch.Tensor], bool]] = None,
    ):
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.min_std = min_std
        self.max_std = max_std
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.min_scaled_compression_ratio = min_scaled_compression_ratio
        self.max_scaled_compression_ratio = max_scaled_compression_ratio
        self.scaled_compression_shape = list(scaled_compression_shape)
        self.min_blurred_compression_ratio = min_blurred_compression_ratio
        self.max_blurred_compression_ratio = max_blurred_compression_ratio
        self.blurred_compression_kernel_size = list(blurred_compression_kernel_size)
        self.blurred_compression_sigma = blurred_compression_sigma
        self.compression_format = compression_format
        self.callable = callable

    def __call__(self, image: torch.Tensor) -> bool:
        if self.min_mean or self.max_mean:
            mean = image.mean()
            if self.min_mean and mean < self.min_mean:
                return False
            if self.max_mean and mean > self.max_mean:
                return False

        if self.min_std or self.max_std:
            std = image.std()
            if self.min_std and std < self.min_std:
                return False
            if self.max_std and std > self.max_std:
                return False

        if self.min_compression_ratio or self.max_compression_ratio:
            ratio = self.calc_compression_ratio(image)
            if self.min_compression_ratio and ratio < self.min_compression_ratio:
                return False
            if self.max_compression_ratio and ratio > self.max_compression_ratio:
                return False

        if self.min_scaled_compression_ratio or self.max_scaled_compression_ratio:
            ratio = self.calc_compression_ratio(
                VF.resize(image, self.scaled_compression_shape, interpolation=VF.InterpolationMode.BICUBIC)
            )
            if self.min_scaled_compression_ratio and ratio < self.min_scaled_compression_ratio:
                return False
            if self.max_scaled_compression_ratio and ratio > self.max_scaled_compression_ratio:
                return False

        if self.min_blurred_compression_ratio or self.max_blurred_compression_ratio:
            ratio = self.calc_compression_ratio(
                VF.gaussian_blur(image, self.blurred_compression_kernel_size, self.blurred_compression_sigma)
            )
            if self.min_blurred_compression_ratio and ratio < self.min_blurred_compression_ratio:
                return False
            if self.max_blurred_compression_ratio and ratio > self.max_blurred_compression_ratio:
                return False

        if self.callable is not None:
            if not self.callable(image):
                return False

        return True

    def calc_compression_ratio(self, image: torch.Tensor) -> float:
        img = VF.to_pil_image(image)
        fp = BytesIO()
        img.save(fp, self.compression_format)
        memory_size = math.prod(image.shape)
        compress_size = fp.tell()
        return compress_size / memory_size
