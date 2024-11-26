from typing import Optional

import pygame
import torch


def tensor_to_pygame_surface(tensor) -> pygame.Surface:
    """
    Mhhh, this currently makes about 50 conversions per second for a 3x1024x1024 image
        with A LOT of cpu
    """
    assert tensor.ndim == 3, f"Got {tensor.shape}"

    tensor: torch.Tensor = (
        (tensor.detach() * 255).clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
    )
    channels = tensor.shape[-1]
    if channels == 3:
        format = "RGB"
    elif channels == 4:
        format = "RGBA"
    else:
        raise NotImplemented(f"Can't handle channels: {channels}")

    return pygame.image.frombuffer(
        tensor.numpy().tobytes(),
        (tensor.shape[1], tensor.shape[0]),
        format,
    )


class BoundingBox:

    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        return f"{self.__class__.__name__}(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_empty(self) -> bool:
        return self.x2 <= self.x1 or self.y2 <= self.y1

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "BoundingBox":
        return BoundingBox(0, 0, tensor.shape[-2], tensor.shape[-1])

    def crop_tensor(self, tensor: torch.Tensor):
        return tensor[..., self.y1: self.y2, self.x1: self.x2]

    def replace(
            self,
            x1: Optional[int] = None,
            y1: Optional[int] = None,
            x2: Optional[int] = None,
            y2: Optional[int] = None,
    ):
        return BoundingBox(
            self.x1 if x1 is None else x1,
            self.y1 if y1 is None else y1,
            self.x2 if x2 is None else x2,
            self.y2 if y2 is None else y2,
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return self.__class__(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def inside(self, other: "BoundingBox") -> "BoundingBox":
        return self.__class__(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )

    def fit_into_at(self, other: "BoundingBox", x: int, y: int) -> Optional["BoundingBox"]:
        """
        make `rect` fit into another rect without overlap
        """
        rect = self
        if x < 0:
            if x <= -rect.width:
                return
            rect = rect.replace(x1=-x)

        if y < 0:
            if y <= -rect.height:
                return
            rect = rect.replace(y1=-y)

        if x + rect.width > other.width:
            if x >= other.width:
                return
            over = x + rect.width - other.width
            rect = rect.replace(x2=rect.x1 + rect.width - over)

        if y + rect.height > other.height:
            if y >= other.height:
                return
            over = y + rect.height - other.height
            rect = rect.replace(y2=rect.y1 + rect.height - over)

        if not rect.is_empty:
            return rect


def fit_tensor_rect(rect: torch.Tensor, x: int, y: int, width: int, height: int) -> Optional[torch.Tensor]:
    """
    make `rect` fit into another tensor with shape [..., :height, :width]
    """
    if x < 0:
        if x <= -rect.shape[-1]:
            return
        rect = rect[..., :, -x:]
        x = 0

    if y < 0:
        if y <= -rect.shape[-2]:
            return
        rect = rect[..., -y:, :]
        y = 0

    if x + rect.shape[-1] > width:
        if x >= width:
            return
        rect = rect[..., :, :-(x + rect.shape[-1] - width)]

    if y + rect.shape[-2] > height:
        if y >= height:
            return
        rect = rect[..., :-(y + rect.shape[-2] - height), :]

    if rect.shape[-1] and rect.shape[-2]:
        return rect

