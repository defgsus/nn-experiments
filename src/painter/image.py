from typing import Tuple

import pygame
import PIL.Image
import torch
import torchvision.transforms.functional as VF

from .util import tensor_to_pygame_surface, BoundingBox


class Image:

    def __init__(self, tensor: torch.Tensor):
        assert tensor.ndim == 3, f"Got {tensor.shape}"
        assert tensor.shape[0] in (3, 4), f"Got {tensor.shape}"
        self.tensor = tensor

    def __repr__(self):
        return f"Image({self.tensor.shape[0]}x{self.tensor.shape[1]}x{self.tensor.shape[2]})"

    @property
    def width(self) -> int:
        return self.tensor.shape[-1]

    @property
    def height(self) -> int:
        return self.tensor.shape[-2]

    @property
    def channels(self) -> int:
        return self.tensor.shape[-3]

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(0, 0, self.width, self.height)

    def crop(self, box: BoundingBox) -> "Image":
        if not (
                0 <= box.x1 <= self.width
            and 0 <= box.x2 <= self.width
            and 0 <= box.y1 <= self.height
            and 0 <= box.y2 <= self.height
        ):
            raise ValueError(f"Invalid box {box} to crop image {self}")

        return Image(box.crop_tensor(self.tensor))

    def to_pygame_surface(self) -> pygame.Surface:
        return tensor_to_pygame_surface(self.tensor)

    def update_pygame_surface(self, surface: pygame.Surface, rect: BoundingBox):
        sbox = BoundingBox(0, 0, *surface.get_size())
        rect = rect.inside(sbox).inside(self.bounding_box)
        tensor = self.tensor[..., rect.y1: rect.y2, rect.x1: rect.x2]
        new_surface = tensor_to_pygame_surface(tensor)
        surface.blit(new_surface, (rect.x1, rect.y1))

    @classmethod
    def from_file(cls, file):
        image = PIL.Image.open(file)
        return cls(VF.to_tensor(image.convert("RGB")))

