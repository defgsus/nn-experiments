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

        for i in range(22):
            box = ColorBoxWidget(
                self, color=QColor(random.randrange(256), random.randrange(256), random.randrange(256)),
            )
            box.setFixedSize(16, 16)
            box.clicked.connect(partial(self.signal_color_changed.emit, box.color))
            self._boxes.append(box)
            lv.addWidget(box)

        lv.addStretch(10)


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