from typing import Optional, Dict

import torch
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


# global lookup
transformations: Dict[str, "TransformBase"] = dict()


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

    NAME: Optional[str] = None
    IS_RANDOM = False
    IS_RESIZE = False
    PARAMS: Optional[dict] = None

    def __init_subclass__(cls, **kwargs):
        assert cls.NAME, f"Must specify {cls.__name__}.NAME"
        assert cls.PARAMS, f"Must specify {cls.__name__}.PARAMS"
        if cls.NAME in transformations:
            raise ValueError(
                f"{cls.__name__}.NAME = '{cls.NAME}' is already defined for {transformations[cls.NAME].__name__}"
            )
        transformations[cls.NAME] = cls

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Scale(TransformBase):
    """
    Scale by a fixed ratio
    """
    NAME = "scale"
    IS_RESIZE = True
    PARAMS = [
        {
            "name": "scale",
            "type": "float",
            "default": 2.,
        },
        INTERPOLATION_PARAMETER,
    ]

    def __init__(self, scale: float, interpolation: str):
        super().__init__()
        self.scale = scale
        self.interpolation = to_torch_interpolation(interpolation)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        new_shape = [max(1, int(s * self.scale)) for s in image.shape[-2:]]

        return VF.resize(
            image,
            size=new_shape,
            interpolation=self.interpolation,
            antialias=self.interpolation != VF.InterpolationMode.NEAREST,
        )


class RandomCrop(TransformBase):
    """
    Crop at a random location.
    """
    NAME = "random_crop"
    IS_RESIZE = True
    IS_RANDOM = True
    PARAMS = [
        {
            "name": "size",
            "type": "int",
            "default": 224,
        },
    ]

    def __init__(self, size: int):
        super().__init__()
        self.transform = VT.RandomCrop(size=size, pad_if_needed=True)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


class RandomAffine(TransformBase):
    """
    torchvision.transforms.RandomAffine
    """
    NAME = "random_affine"
    IS_RESIZE = True
    IS_RANDOM = True
    PARAMS = [
        {
            "name": "degrees_min",
            "type": "float",
            "default": -5.,
            "min": -360,
            "max": 360,
        },
        {
            "name": "degrees_max",
            "type": "float",
            "default": 5.,
            "min": -360,
            "max": 360,
        },
        {
            "name": "translate_x",
            "type": "float",
            "default": 0.,
            "min": -2.,
            "max": 2.,
        },
        {
            "name": "translate_y",
            "type": "float",
            "default": 0.,
            "min": -2.,
            "max": 2.,
        },
        {
            "name": "scale_min",
            "type": "float",
            "default": 1.,
            "min": 0,
            "max": 100.,
        },
        {
            "name": "scale_max",
            "type": "float",
            "default": 1.,
            "min": 0,
            "max": 100.,
        },
        {
            "name": "shear_min",
            "type": "float",
            "default": 0.,
            "min": -360,
            "max": 360.,
        },
        {
            "name": "shear_max",
            "type": "float",
            "default": 0.,
            "min": -360,
            "max": 360.,
        },
        INTERPOLATION_PARAMETER
    ]

    def __init__(
            self,
            degrees_min: float,
            degrees_max: float,
            translate_x: float,
            translate_y: float,
            scale_min: float,
            scale_max: float,
            shear_min: float,
            shear_max: float,
            interpolation: str,
    ):
        super().__init__()
        self.transform = VT.RandomAffine(
            degrees=(degrees_min, degrees_max),
            translate=None if translate_x == 0 and translate_y == 0 else (translate_x, translate_y),
            scale=None if scale_min == 1 and scale_max == 1 else (scale_min, scale_max),
            shear=None if shear_min == 0 and shear_max == 0 else (shear_min, shear_max, shear_min, shear_max),
            interpolation=to_torch_interpolation(interpolation),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)
