import torch

from .base import *


class RandomTransformBase(TransformBase):
    CLASS = "random"
    IS_RANDOM = True


class Noise(RandomTransformBase):
    """
    Crop at a random location.
    """
    NAME = "noise"
    PARAMS = [
        *TransformBase.PARAMS,
        {
            "name": "amount",
            "type": "float",
            "default": .1,
        },
    ]

    def __init__(
            self,
            amount: int,
    ):
        super().__init__()
        self.amount = amount

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image + torch.randn_like(image) * self.amount).clamp(0, 1)

