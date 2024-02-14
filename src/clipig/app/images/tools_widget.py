from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage, LImageLayer
from . import image_tools


class ImageToolsWidget(QWidget):

    signal_tool_changed = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        lh = QHBoxLayout(self)
        self.setLayout(lh)

        for klass in image_tools.image_tools.values():
            butt = QToolButton(self)
            butt.setText(klass.NAME)
            butt.clicked.connect(partial(self.set_tool, klass.NAME))
            lh.addWidget(butt)

    def set_tool(self, tool_name: str):
        if image_tools.image_tools.get(tool_name):
            self.signal_tool_changed.emit(tool_name)

