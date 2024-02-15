import random
from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Dict

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from . import image_tools


class ColorPaletteWidget(QWidget):

    signal_color_changed = pyqtSignal(QColor)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._boxes: List[ColorBoxWidget] = []

        self._create_widgets()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lv)

        lg = QGridLayout()
        lg.setContentsMargins(0, 0, 0, 0)
        lg.setSpacing(0)
        lv.addLayout(lg)
        for i, row in enumerate(self.create_color_grid()):
            for j, col in enumerate(row):
                box = ColorBoxWidget(self, color=col)
                box.setFixedSize(16, 16)
                box.clicked.connect(partial(self.signal_color_changed.emit, box.color))
                self._boxes.append(box)
                lg.addWidget(box, i, j)

        lv.addStretch(10)

    def create_color_grid(self) -> List[List[QColor]]:
        base_colors = [
            (.5, .5, .5),
            (1, 0, 0),
            (1, .5, 0),
            (1, 1, 0),
            (.5, 1, 0),
            (0, 1, 0),
            (0, 1, .7),
            (0, 1, 1),
            (0, .6, 1),
            (0, 0, 1),
            (.5, 0, 1),
            (1, 0, 1),
            (1, 0, .5),
        ]
        def _to_color(col, add: float):
            col = [
                c * (1. + add) + add * .7
                for c in col
            ]
            col = [max(0, min(255, int(c * 255))) for c in col]
            return QColor(col[0], col[1], col[2])

        color_grid = []
        for c in base_colors:
            row = [
                _to_color(c, -.5),
                _to_color(c, -.3),
                _to_color(c, 0),
                _to_color(c, .9),
            ]
            color_grid.append(row)

        return color_grid


class ColorBoxWidget(QWidget):

    clicked = pyqtSignal()

    def __init__(self, *args, color: QColor, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setPen(Qt.GlobalColor.black)
        painter.setBrush(self.color)
        painter.drawRect(self.rect())

    def mousePressEvent(self, event: QMouseEvent):
        self.clicked.emit()