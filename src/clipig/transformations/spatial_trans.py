from .base import *
from src.algo.wangtiles import *


class SpatialTransformBase(TransformBase):
    CLASS = "spatial"


class Scale(SpatialTransformBase):
    """
    Scale by a fixed ratio
    """
    NAME = "scale"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "scale",
            "type": "float",
            "default": 1.,
            "min": 0,
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


class Repeat(SpatialTransformBase):
    """
    Repeat the image in x and y direction
    """
    NAME = "repeat"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "repeat_xy",
            "type": "int2",
            "default": [2, 2],
            "min": [1, 1],
            "max": [100, 100],
        },
    ]

    def __init__(self, repeat_xy: Tuple[int]):
        super().__init__()
        self.repeat_xy = repeat_xy

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.repeat(1, self.repeat_xy[1], self.repeat_xy[0])


class Padding(SpatialTransformBase):
    """
    Pad individual edges of the image
    """
    NAME = "padding"
    IS_RESIZE = True
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "pad_left",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "pad_top",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "pad_right",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "pad_bottom",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 2**16,
        },
        {
            "name": "padding_mode",
            "type": "select",
            "choices": ["constant", "edge", "reflect", "symmetric"],
            "default": "constant",
        }
    ]

    def __init__(
            self,
            pad_left: int,
            pad_top: int,
            pad_right: int,
            pad_bottom: int,
            padding_mode: str,
    ):
        super().__init__()
        self.pad_left = pad_left
        self.pad_top = pad_top
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom
        self.padding_mode = padding_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return VF.pad(
            image,
            [self.pad_left, self.pad_top, self.pad_right, self.pad_bottom],
            padding_mode=self.padding_mode,
        )


class Blur(SpatialTransformBase):
    NAME = "blur"
    PARAMS = [
        *SpatialTransformBase.PARAMS,
        {
            "name": "kernel_size",
            "type": "int2",
            "default": [3, 3],
            "min": [1, 1],
        },
        {
            "name": "sigma",
            "type": "float2",
            "default": [1., 1.],
            "min": [0., 0.],
        },
        {
            "name": "mix",
            "type": "float",
            "default": 1.,
            "min": 0.,
            "max": 1.,
        },
    ]

    def __init__(
            self,
            kernel_size: Tuple[int, int],
            sigma: Tuple[float, float],
            mix: float,
    ):
        super().__init__()
        self.sigma = sigma
        self.mix = mix
        self.kernel_size = [
            max(1, k+1 if k % 2 == 0 else k)
            for k in reversed(kernel_size)
        ]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        blurred = VF.gaussian_blur(image, self.kernel_size, self.sigma)
        if self.mix != 1.:
            blurred = image * (1. - self.mix) + self.mix * blurred
        return blurred

