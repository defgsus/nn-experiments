import math
from typing import Tuple, Dict, Optional, Type, List

import torch
import torch.nn as nn
import torchvision.transforms.functional as VF

from src.util.image import *


source_models: Dict[str, Type["SourceModelBase"]] = {}


class SourceModelBase(nn.Module):

    NAME: Optional[str] = None
    IS_AUTOENCODER: bool = False
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

    def clear(self):
        raise NotImplementedError

    def randomize(self, mean: float, std: float):
        raise NotImplementedError

    def set_image(self, image: torch.Tensor):
        raise NotImplementedError


def fit_image(image: torch.Tensor, shape: Tuple[int, int, int], dtype: torch.dtype):
    if image.shape[-2:] != torch.Size(shape[-2:]):
        image = VF.resize(image, shape[-2:], VF.InterpolationMode.BICUBIC, antialias=True)
    image = set_image_channels(image, shape[0])
    image = set_image_dtype(image, dtype)
    return image
