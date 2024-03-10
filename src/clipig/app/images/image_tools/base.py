import dataclasses
import math
from copy import deepcopy
from typing import Tuple, Dict, Optional, Type, List

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..limage import LImage, LImageLayer


image_tools: Dict[str, Type["ImageToolBase"]] = {}


class MouseEvent:

    Hover = "hover"
    Press = "press"
    Release = "release"
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

    def __init__(self, limage: LImage, config: dict):
        self.limage = limage
        self.config = deepcopy(config)

        for param in self.PARAMS:
            if param["name"] not in self.config:
                self.config[param["name"]] = param["default"]

    @property
    def layer(self):
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

    def paint(self, painter: QPainter, event: QPaintEvent):
        pass

    def mouse_event(self, event: MouseEvent):
        pass

    def config_changed(self):
        pass
