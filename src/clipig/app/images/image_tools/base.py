import dataclasses
import math
from copy import deepcopy
from typing import Tuple, Dict, Optional, Type, List

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..limage import LImage, LImageLayer
# from ..limage_canvas_widget import LImageCanvasWidget


image_tools: Dict[str, Type["ImageToolBase"]] = {}


class MouseEvent:

    Hover = "hover"
    Press = "press"         # begin of click
    Release = "release"     # end of click
    Drag = "drag"
    Click = "click"

    def __init__(
            self,
            type: str,
            x: int,
            y: int,
            button: Qt.MouseButton,
    ):
        self.type = type
        self.x = x
        self.y = y
        self.button = button

    def __repr__(self):
        return f"{self.__class__.__name__}(type={repr(self.type)}, x={self.x}, y={self.y}, button={self.button})"

    @property
    def pos(self) -> Tuple[int, int]:
        return self.x, self.y


class ImageToolBase:

    NAME: str = None
    PARAMS: List[dict] = []

    def __init_subclass__(cls, **kwargs):
        assert cls.NAME, f"Must specify {cls.__name__}.NAME"
        # assert cls.PARAMS, f"Must specify {cls.__name__}.PARAMS"
        if cls.NAME in image_tools and cls != image_tools[cls.NAME]:
            print(image_tools[cls.NAME], cls)
            raise ValueError(
                f"{cls.__name__}.NAME = '{cls.NAME}' is already defined for {image_tools[cls.NAME].__name__}"
            )
        image_tools[cls.NAME] = cls

    def __init__(self, limage: LImage, config: dict, project: "ProjectWidget"):
        from ...project.project_widget import ProjectWidget
        self.limage = limage
        self.config = deepcopy(config)
        self.project: ProjectWidget = project

        for param in self.PARAMS:
            if param["name"] not in self.config:
                self.config[param["name"]] = param["default"]

    @property
    def layer(self) -> Optional[LImageLayer]:
        return self.limage.selected_layer

    def set_config(self, config: dict):
        self.config = deepcopy(config)

        for param in self.PARAMS:
            if param["name"] not in self.config:
                self.config[param["name"]] = param["default"]

        self.config_changed()

    # --- stuff to override ---

    def add_menu_actions(self, menu: QMenu, project: "ProjectWidget"):
        pass

    def paint(self, painter: QPainter, event: QPaintEvent, canvas: "LImageCanvasWidget"):
        pass

    def mouse_event(self, event: MouseEvent):
        pass

    def config_changed(self):
        pass
