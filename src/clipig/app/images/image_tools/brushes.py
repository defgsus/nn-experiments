from typing import Tuple, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ..limage import LImage


class Brush:

    PARAMS = [
        {
            "name": "type",
            "type": "select",
            "default": "rectangle",
            "choices": ["rectangle"],
        },
        {
            "name": "size",
            "type": "float2",
            "default": [10., 10.],
            "min": [0., 0.],
        },
        {
            "name": "color",
            "type": "int3",
            "default": [128, 128, 128],
            "min": [0, 0, 0],
            "max": [255, 255, 255],
        },
        {
            "name": "alpha",
            "type": "float",
            "default": 1.,
            "min": 0.,
            "max": 1.,
        },
    ]

    def __init__(
            self,
            type: str,
            size: Union[float, Tuple[float, float]],
            color: Tuple[int, int, int],
            alpha: float = 1.,
    ):
        self.type = type
        self.size = (size, size) if isinstance(size, (int, float)) else tuple(size)
        self.color = tuple(color)
        self.alpha = alpha

    def apply(
            self,
            limage: LImage,
            painter: QPainter,
            pos: Tuple[int, int],
            delta: Tuple[int, int] = (0, 0),
    ):
        painter.setOpacity(self.alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*self.color))

        if self.type == "rectangle":
            s = self.size
            rect = QRectF(pos[0] - s[0] / 2, pos[1] - s[1] / 2, s[0], s[1])
            for rect, offset in limage.pixel_rects_to_image_rects([rect]):
                painter.drawRect(rect)

        painter.setOpacity(1.)
