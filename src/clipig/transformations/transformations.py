from .base import *


class Scale(TransformBase):
    """
    Scale by a fixed ratio
    """
    NAME = "scale"
    IS_RESIZE = True
    PARAMS = [
        *TransformBase.PARAMS,
        {
            "name": "scale",
            "type": "float",
            "default": 2.,
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


class RandomCrop(TransformBase):
    """
    Crop at a random location.
    """
    NAME = "random_crop"
    IS_RESIZE = True
    IS_RANDOM = True
    PARAMS = [
        *TransformBase.PARAMS,
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
        *TransformBase.PARAMS,
        {
            "name": "degrees_min_max",
            "type": "float2",
            "default": [-5., 5.],
            "step": [5., 5.],
            "min": [-360., -360],
            "max": [360., 360.],
        },
        {
            "name": "translate_xy",
            "type": "float2",
            "default": [0., 0.],
            "min": [-2., -2.],
            "max": [2., 2.],
        },
        {
            "name": "scale_min_max",
            "type": "float2",
            "default": [1., 1.],
            "min": [0., 0.],
            "max": [100., 100.],
        },
        {
            "name": "shear_min_max",
            "type": "float2",
            "default": [0., 0.],
            "step": [5., 5.],
            "min": [-360., -360.],
            "max": [360., 360.],
        },
        INTERPOLATION_PARAMETER
    ]

    def __init__(
            self,
            degrees_min_max: Tuple[float, float],
            translate_xy: Tuple[float, float],
            scale_min_max: Tuple[float, float],
            shear_min_max: Tuple[float, float],
            interpolation: str,
    ):
        super().__init__()
        self.transform = VT.RandomAffine(
            degrees=degrees_min_max,
            translate=None if translate_xy == (0, 0) else translate_xy,
            scale=None if scale_min_max == (1, 1) else scale_min_max,
            shear=None if shear_min_max == (0, 0) else (*shear_min_max, *shear_min_max),
            interpolation=to_torch_interpolation(interpolation),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


class Repeat(TransformBase):
    """
    Repeat the image in x and y direction
    """
    NAME = "repeat"
    IS_RESIZE = True
    IS_RANDOM = True
    PARAMS = [
        *TransformBase.PARAMS,
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
