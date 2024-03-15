from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage


class LImageCanvasControls(QWidget):

    def __init__(self, canvas):
        super().__init__(canvas)

        from .limage_canvas_widget import LImageCanvasWidget
        self.canvas: LImageCanvasWidget = canvas

        self._ignore_zoom_bar = False

        self._create_widgets()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        lh = QHBoxLayout()
        lv.addLayout(lh)

        self.zoom_bar = QScrollBar(Qt.Horizontal, self)
        lh.addWidget(self.zoom_bar)
        self.zoom_bar.setStatusTip(self.tr("zoom"))
        self.zoom_bar.setRange(1, 1500)
        self.zoom_bar.setValue(100)
        self.zoom_bar.valueChanged.connect(self._zoom_bar_changed)
        self.zoom_bar.setToolTip(self.tr("zoom"))

        for zoom in (50, 100, 200, 300, 500, 1000, 1500):
            b = QPushButton(self.tr(f"{zoom}%"))
            if zoom == 100:
                font = b.font()
                font.setBold(True)
                b.setFont(font)
            b.setToolTip(self.tr("set zoom to {zoom}%").format(zoom=zoom))
            lh.addWidget(b)
            b.clicked.connect(partial(self.set_zoom, zoom))

        lh.addSpacing(10)

        self.select_background = QComboBox(self)
        lh.addWidget(self.select_background)
        self.select_background.setToolTip(self.tr("background"))
        self.select_background.addItems([
            "cross", "black", "gray", "white"
        ])
        self.select_background.currentTextChanged.connect(self.set_background)

    def set_zoom(self, z: int):
        self.canvas.set_zoom(z)
        try:
            self._ignore_zoom_bar = True
            self.zoom_bar.setValue(z)
        finally:
            self._ignore_zoom_bar = False

    def _zoom_bar_changed(self, value):
        if not self._ignore_zoom_bar:
            self.set_zoom(value)

    def set_background(self, mode: str):
        self.canvas.set_background(mode)
