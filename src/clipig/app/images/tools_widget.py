from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage, LImageLayer
from . import image_tools
from ..parameter_widget import ParameterWidget
from .tool_buttons import ImageToolButtons


class ImageToolsWidget(QWidget):

    signal_tool_changed = pyqtSignal(str, dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._create_widgets()
        self.set_tool("select")

    def _create_widgets(self):
        lh = QHBoxLayout(self)
        self.setLayout(lh)

        self.tool_buttons = ImageToolButtons(self)
        self.tool_buttons.signal_tool_changed.connect(self._tool_changed)
        lh.addWidget(self.tool_buttons)

        lh.addStretch(10)

        self.config_widget = ImageToolsConfigWidget(self)
        lh.addWidget(self.config_widget)
        self.config_widget.signal_config_changed.connect(self._config_changed)

    def set_tool(self, tool_name: str, emit: bool = True):
        if image_tools.image_tools.get(tool_name):
            self.tool_buttons.set_tool(tool_name)
            self.config_widget.set_tool(tool_name)

            if emit:
                self._config_changed()

    def set_color(self, color: Union[Tuple[int, int, int], QColor], emit: bool = True):
        if isinstance(color, QColor):
            color = (color.red(), color.green(), color.blue())
        self.config_widget.set_value("color", color, emit=emit)

    def _tool_changed(self):
        tool_name = self.tool_buttons.current_tool
        self.config_widget.set_tool(tool_name)
        self.signal_tool_changed.emit(tool_name, self.config_widget.get_values())

    def _config_changed(self):
        tool_name = self.tool_buttons.current_tool
        self.signal_tool_changed.emit(tool_name, self.config_widget.get_values())


class ImageToolsConfigWidget(QWidget):

    signal_config_changed = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param_widgets: Dict[str, ParameterWidget] = {}
        self._create_widgets()

    def get_values(self) -> dict:
        return {
            key: widget.get_value()
            for key, widget in self._param_widgets.items()
        }

    def set_value(self, name: str, value, emit: bool = True):
        if name in self._param_widgets:
            self._param_widgets[name].set_value(value, emit=emit)

    def _create_widgets(self):
        lv = QHBoxLayout(self)
        self.setLayout(lv)

        self.params_widget = QWidget(self)
        lv.addWidget(self.params_widget)
        self._layout = lv

    def set_tool(self, tool_name: str):
        self.tool_name = tool_name

        self.params_widget.deleteLater()
        self.params_widget = QWidget(self)
        self._layout.addWidget(self.params_widget)

        klass = image_tools.image_tools[self.tool_name]

        lv = QVBoxLayout(self.params_widget)
        lv.setContentsMargins(0, 0, 0, 0)

        self._param_widgets.clear()
        for param in klass.PARAMS:
            self._param_widgets[param["name"]] = widget = ParameterWidget(param, self)
            lv.addWidget(widget)
            widget.signal_value_changed.connect(lambda : self.signal_config_changed.emit())
