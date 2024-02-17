from .base import *


class ValueTransformBase(TransformBase):
    CLASS = "value"


class Multiplication(ValueTransformBase):

    NAME = "multiplication"
    PARAMS = [
        *ValueTransformBase.PARAMS,
        {
            "name": "multiply",
            "type": "float",
            "default": 1.,
        },
        {
            "name": "add",
            "type": "float",
            "default": 0.,
        },
    ]

    def __init__(self, multiply: float, add: float):
        super().__init__()
        self.multiply = multiply
        self.add = add

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.multiply + self.add


class Saturation(ValueTransformBase):

    NAME = "saturation"
    PARAMS = [
        *ValueTransformBase.PARAMS,
        {
            "name": "saturation",
            "type": "float",
            "default": 1.,
            "min": 0.0,
        },
    ]

    def __init__(self, saturation: float):
        super().__init__()
        self.saturation = saturation

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return VF.adjust_saturation(image, self.saturation)


class Quantization(ValueTransformBase):

    NAME = "quantization"
    PARAMS = [
        *ValueTransformBase.PARAMS,
        {
            "name": "steps",
            "type": "float",
            "default": .5,
            "min": 0.0,
        },
    ]

    def __init__(self, steps: float):
        super().__init__()
        self.steps = max(0.0000001, steps)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return torch.floor(image / self.steps) * self.steps
