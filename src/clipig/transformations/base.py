import math
import random
from typing import Optional, Dict, Type, Tuple, List, Union

import torch
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


# global lookup
transformations: Dict[str, Type["TransformBase"]] = dict()


def to_torch_interpolation(interpolation: str):
    return {
        "none": VF.InterpolationMode.NEAREST,
        "bilinear": VF.InterpolationMode.BILINEAR,
        "bicubic": VF.InterpolationMode.BICUBIC,
    }.get(interpolation, VF.InterpolationMode.NEAREST)


INTERPOLATION_PARAMETER = {
    "name": "interpolation",
    "type": "select",
    "default": "none",
    "choices": ["none", "bilinear", "bicubic"],
}


class TransformBase:

    CLASS: str = None
    NAME: str = None
    IS_RANDOM = False
    IS_RESIZE = False
    PARAMS: List[dict] = [
        {
            "name": "active",
            "type": "bool",
            "default": True,
        }
    ]

    def __init_subclass__(cls, **kwargs):
        if not cls.__name__.endswith("Base"):
            assert cls.CLASS, f"Must specify {cls.__name__}.CLASS"
            assert cls.NAME, f"Must specify {cls.__name__}.NAME"
            assert cls.PARAMS, f"Must specify {cls.__name__}.PARAMS"
            if cls.NAME in transformations:
                raise ValueError(
                    f"{cls.__name__}.NAME = '{cls.NAME}' is already defined for {transformations[cls.NAME].__name__}"
                )
            transformations[cls.NAME] = cls

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
