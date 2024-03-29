from functools import partial
from pathlib import Path
from copy import deepcopy
from typing import Optional, List, Union, Dict

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage
from .image_tools import ImageToolBase, MouseEvent


class LImageCanvasWidget(QWidget):

    signal_mouse_event = pyqtSignal(MouseEvent)

    def __init__(self, parent):
        super().__init__(parent)

        self.limage: Optional[LImage] = None
        self._zoom = 100
        self._size = (0, 0)
        self._background = "cross"
        self._visible: Dict[str, bool] = {}
        self._tool: Optional[ImageToolBase] = None
        self._last_mouse_press_button = None

    @property
    def zoom(self):
        return self._zoom

    def get_settings(self) -> dict:
        return {
            "zoom": self._zoom,
            "background": self._background,
            "visible": self._visible,
        }

    def set_settings(self, settings: dict):
        settings = deepcopy(settings)
        self._background = settings["background"]
        self.set_zoom(settings["zoom"])
        self._visible = settings.get("visible", {})

    def set_zoom(self, z):
        self._zoom = z
        self._update_size()

    def _update_size(self):
        self.setFixedSize(
            max(1, int(self._size[0] * self.zoom / 100)),
            max(1, int(self._size[1] * self.zoom / 100)),
        )

    def set_background(self, mode: str):
        self._background = mode
        self.update()

    def is_visible(self, name: str) -> bool:
        return bool(self._visible.get(name))

    def set_visible(self, name: str, visible: bool = True):
        self._visible[name] = bool(visible)
        self.update()

    def set_limage(self, image: LImage):
        self.limage = image
        rect = self.limage.rect()
        self._size = (rect.right(), rect.bottom())
        self.set_zoom(self.zoom)

        self.limage.get_model().dataChanged.connect(lambda *args, **kwargs: self.update())
        self.limage.get_model().modelReset.connect(lambda *args, **kwargs: self.update())

    def set_tool(self, tool: Optional[ImageToolBase]):
        self._tool = tool

    def paintEvent(self, event: QPaintEvent):
        if self.limage is not None:
            rect = self.limage.rect(for_ui=True)
            size = (rect.right(), rect.bottom())
            if size != self._size:
                self._size = size
                self._update_size()

        painter = QPainter(self)

        trans = QTransform()
        trans.scale(self.zoom / 100., self.zoom / 100.)
        painter.setTransform(trans)

        painter.setPen(Qt.NoPen)

        brush = QBrush(QColor(0, 0, 0))
        if self._background == "cross":
            brush = QBrush(QColor(128, 128, 128), Qt.BrushStyle.DiagCrossPattern)
        elif self._background == "black":
            pass
        elif self._background == "white":
            brush = QBrush(QColor(255, 255, 255))
        elif self._background == "red":
            brush = QBrush(QColor(255, 0, 0))
        elif self._background == "green":
            brush = QBrush(QColor(0, 255, 0))
        elif self._background == "blue":
            brush = QBrush(QColor(0, 0, 255))
        painter.setBrush(brush)
        painter.drawRect(0, 0, *self._size)

        if self.limage:
            self.limage.paint(painter, for_ui=True)

        if self.limage.tiling and not self.limage.ui_settings.project_random_tiling_map:
            if self.is_visible("tiling_grid"):
                self.limage.paint_tiling_grid(painter)

            if self.is_visible("tiling"):
                self.limage.paint_tiling(painter)

        if self._tool:
            self._tool.paint(painter, event, self)

    def create_controls(self):
        from .limage_canvas_controls import LImageCanvasControls

        widget = LImageCanvasControls(self)
        return widget

    def _convert_mouse_event(
            self,
            type: str,
            event: QMouseEvent,
            button: Optional[int] = None,
    ):
        return MouseEvent(
            type=type,
            x=int(event.x() * 100 / self._zoom),
            y=int(event.y() * 100 / self._zoom),
            button=event.button() if button is None else button,
        )

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_press_button = event.button()
        event = self._convert_mouse_event(MouseEvent.Press, event)
        self.signal_mouse_event.emit(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        event = self._convert_mouse_event(MouseEvent.Release, event, button=self._last_mouse_press_button)
        self._last_mouse_press_button = None
        self.signal_mouse_event.emit(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.signal_mouse_event.emit(MouseEvent(
            type=MouseEvent.Drag,
            x=int(event.x() * 100 / self._zoom),
            y=int(event.y() * 100 / self._zoom),
            button=self._last_mouse_press_button,
        ))
