from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage


class LImageCanvasWidget(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.limage: Optional[LImage] = None
        self._zoom = 100
        self._size = (0, 0)
        self._background = "cross"

    @property
    def zoom(self):
        return self._zoom

    def set_zoom(self, z):
        self._zoom = z
        self._update_size()

    def _update_size(self):
        self.setFixedSize(
            max(1, self._size[0] * self.zoom),
            max(1, self._size[1] * self.zoom),
        )

    def set_background(self, mode: str):
        self._background = mode
        self.update()

    def set_limage(self, image: LImage):
        self.limage = image
        rect = self.limage.rect()
        self._size = (rect.right(), rect.bottom())
        self.set_zoom(self.zoom)

        self.limage.get_model().dataChanged.connect(lambda *args, **kwargs: self.update())
        self.limage.get_model().modelReset.connect(lambda *args, **kwargs: self.update())

    def paintEvent(self, event: QPaintEvent):
        if self.limage is not None:
            rect = self.limage.rect()
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
            self.limage.paint(painter)

    def create_controls(self):
        from .limage_canvas_controls import LImageCanvasControls

        widget = LImageCanvasControls(self)
        return widget
