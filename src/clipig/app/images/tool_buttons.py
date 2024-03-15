from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Dict

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from . import image_tools


class ImageToolButtons(QWidget):

    signal_tool_changed = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tool_name = "select"
        self._ignore_tool_change = True
        self._buttons: Dict[str, QToolButton] = {}

        self._create_widgets()
        self.set_tool(self._tool_name)

    @property
    def current_tool(self) -> str:
        return self._tool_name

    def _create_widgets(self):
        lg = QGridLayout(self)
        lg.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lg)

        x, y = 0, 0
        for klass in image_tools.image_tools.values():
            self._buttons[klass.NAME] = butt = QToolButton(self)
            butt.setText(klass.NAME)
            butt.clicked.connect(partial(self._tool_click, klass.NAME))
            lg.addWidget(butt, y, x)
            x += 1
            if x == 2:
                x, y = 0, y + 1

        # lh.addStretch(10)

    def set_tool(self, tool_name: str):
        if image_tools.image_tools.get(tool_name):
            self._update_tool(tool_name)

    def _tool_click(self, tool_name: str):
        self._update_tool(tool_name)
        self.signal_tool_changed.emit(tool_name)

    def _update_tool(self, tool_name: str):
        self._buttons[self._tool_name].setDown(False)
        self._tool_name = tool_name
        self._buttons[self._tool_name].setDown(True)
