import dataclasses
import math
from typing import Tuple, Dict, Optional, Type, List

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..limage import LImage, LImageLayer


image_tools: Dict[str, Type["ImageToolBase"]] = {}


@dataclasses.dataclass
class MouseEvent:
    type: str
    x: int
    y: int
    button: Qt.MouseButton

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


MouseEvent.Hover = "hover"
MouseEvent.Press = "pressr"
MouseEvent.Release = "release"
MouseEvent.Drag = "drag"
MouseEvent.Click = "click"


class ImageToolBase:

    NAME: str = None
    PARAMS: List[dict] = []

    def __init_subclass__(cls, **kwargs):
        assert cls.NAME, f"Must specify {cls.__name__}.NAME"
        # assert cls.PARAMS, f"Must specify {cls.__name__}.PARAMS"
        if cls.NAME in image_tools:
            raise ValueError(
                f"{cls.__name__}.NAME = '{cls.NAME}' is already defined for {image_tools[cls.NAME].__name__}"
            )
        image_tools[cls.NAME] = cls

    def __init__(self, limage: LImage):
        self.limage = limage

    @property
    def layer(self):
        return self.limage.selected_layer

    def mouse_event(self, event: MouseEvent):
        pass


class SelectionTool(ImageToolBase):

    NAME = "select"


class MoveTool(ImageToolBase):

    NAME = "move"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_pos = None
        self._mouse_pos = None

    def mouse_event(self, event: MouseEvent):
        if event.type == MouseEvent.Press:
            self._layer_pos = self.layer.position
            self._mouse_pos = event.pos

        if event.type == MouseEvent.Drag and self._layer_pos:
            self.layer.set_position((
                self._layer_pos[0] + event.x - self._mouse_pos[0],
                self._layer_pos[1] + event.y - self._mouse_pos[1],
            ))
