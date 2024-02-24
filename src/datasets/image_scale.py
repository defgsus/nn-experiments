from typing import Union, Generator, Optional, Callable, Any, Dict, Iterable, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from .base_iterable import BaseIterableDataset


class ImageScaleIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
            scales: Union[Iterable[float], Callable[[Tuple[int, int]], Iterable[float]]],
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
            interpolation: VT.InterpolationMode = VT.InterpolationMode.BILINEAR,
            with_scale: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.scales = scales if callable(scales) else tuple(scales)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        self.with_scale = bool(with_scale)

    def __iter__(self):
        for data in self.dataset:
            is_tuple = isinstance(data, (tuple, list))
            if is_tuple:
                image = data[0]
            else:
                image = data

            scales = self.scales
            if callable(scales):
                scales = scales(image.shape[-2:])

            for scale in scales:

                scaled_shape = tuple(int(s * scale) for s in image.shape[-2:])

                if self.min_size is not None and any(s < self.min_size for s in scaled_shape):
                    continue
                if self.max_size is not None and any(s > self.max_size for s in scaled_shape):
                    continue

                scaled_image = VF.resize(
                    image, scaled_shape,
                    interpolation=self.interpolation,
                    antialias=self.interpolation != VF.InterpolationMode.NEAREST,
                )

                if self.with_scale:
                    if is_tuple:
                        yield scaled_image, scale, *data[1:]
                    else:
                        yield scaled_image, scale
                else:
                    if is_tuple:
                        yield scaled_image, *data[1:]
                    else:
                        yield scaled_image
