from pathlib import Path

import yaml

from .base import *
from ..source_models.util import load_model_from_yaml, PROCESS_PATH, get_full_yaml_filename
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
            "default": None,  # will be filled at import time
            "choices": [None],
        },
        {
            "name": "mix",
            "type": "float",
            "default": 1.,
            "min": 0.0,
            "max": 1.0,
        },
        {
            "name": "overlap",
            "type": "int2",
            "default": [7, 7],
            "min": [0, 0],
            "max": [1024, 1024],
        },
        {
            "name": "cut_away",
            "type": "int2",
            "default": [0, 0],
            "min": [0, 0],
            "max": [1024, 1024],
        },
        {
            "name": "padding_mode",
            "type": "select",
            "default": "constant",
            "choices": ["constant", "reflect", "replicate", "circular"],
        },
    ]

    def __init__(
            self,
            model: str,
            mix: float,
            overlap: Tuple[int, int] = (0, 0),
            cut_away: Tuple[int, int] = (0, 0),
            padding_mode: str = "constant",
    ):
        super().__init__()
        self.model, self.model_config = load_model_from_yaml(PROCESS_PATH / f"{model}.yaml")
        self.model.eval()
        self.mix = mix
        self.overlap = overlap
        self.cut_away = cut_away
        self.padding_mode = padding_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-3] == 4:
            alpha = image[3:]
            image = image[:3]
        else:
            alpha = None

        processed = map_image_patches(
            image=image,
            function=lambda x: self.model(x).clamp(0, 1),
            patch_size=self.model_config["shape"][-2:],
            overlap=tuple(reversed(self.overlap)),
            cut_away=tuple(reversed(self.cut_away)),
            batch_size=64,
            padding_mode=self.padding_mode,
        )
        if self.mix != 1.:
            processed = image * (1. - self.mix) + self.mix * processed

        if alpha is not None:
            processed = torch.concat([processed, alpha])

        return processed


def _get_denoising_modules():
    directory = Path(__file__).resolve().parent.parent / "models" / "process"

    models = []
    for f in directory.glob("*.yaml"):
        with f.open() as fp:
            data = yaml.safe_load(fp)
            if get_full_yaml_filename(data["checkpoint"], directory).exists():
                models.append(f.name[:-5])
            else:
                print(f"Missing checkpoint, skipping model {f.name}")

    assert models, f"No models found in {directory}"

    param = next(filter(lambda p: p["name"] == "model", Denoising.PARAMS))
    param["choices"] = sorted(models)
    param["default"] = param["choices"][0]


_get_denoising_modules()
