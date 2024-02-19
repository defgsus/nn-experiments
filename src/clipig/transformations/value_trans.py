from .base import *
from ..source_models.util import load_model_from_yaml, PROCESS_PATH
from src.util.image import map_image_patches


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


class Denoising(ValueTransformBase):

    NAME = "denoising"
    PARAMS = [
        *ValueTransformBase.PARAMS,
        {
            "name": "model",
            "type": "select",
            "default": "denoiser-conv-32x32-750k",
            "choices": ["denoiser-conv-32x32-750k"],
        },
        {
            "name": "mix",
            "type": "float",
            "default": 1.,
            "min": 0.0,
            "max": 1.0,
        },
    ]

    def __init__(
            self,
            model: str,
            mix: float,
    ):
        super().__init__()
        self.model, self.model_config = load_model_from_yaml(PROCESS_PATH / f"{model}.yaml")
        self.mix = mix

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        processed = map_image_patches(
            image=image,
            function=lambda x: self.model(x).clamp(0, 1),
            patch_size=self.model_config["shape"][-2:],
            overlap=0,
            batch_size=64,
        )
        if self.mix == 1.:
            return processed
        return image * (1. - self.mix) + self.mix * processed

