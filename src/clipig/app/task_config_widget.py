from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import yaml

from .util import image_to_qimage
from .image_widget import ImageWidget
from ..clipig_worker import ClipigWorker



class TaskConfigWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with (Path(__file__).resolve().parent / "task_parameters.yaml").open() as fp:
            self.parameters = yaml.safe_load(fp)

        self._create_widgets()
        self.set_values({})

    def _create_widgets(self):
        lv = QVBoxLayout(self)

        for param in self.parameters["base"]:

            self._create_param_widget(param)
            widget = param["widget"]

            lh = QHBoxLayout()
            lv.addLayout(lh)

            lh.addWidget(QLabel(param["name"], self))
            lh.addWidget(widget)

    def _create_param_widget(self, param: dict):
        param["get_value"] = lambda w: w.value()
        param["set_value"] = lambda w, v: w.setValue(v)

        if param["type"] == "str":
            widget = QLineEdit(self)
            param["get_value"] = lambda w: w.text()
            param["set_value"] = lambda w, v: w.setText(v)

        elif param["type"] == "int":
            widget = QSpinBox(self)
            widget.setRange(param.get("min", 0), param.get("max", 2**24))

        elif param["type"] == "float":
            widget = QDoubleSpinBox(self)
            widget.setRange(param.get("min", 0), param.get("max", 2**24))
            widget.setSingleStep(param.get("step", 0.1))

        else:
            raise ValueError(f"Unhandled type '{param['type']} in parameter: {param}")

        param["widget"] = widget

    # def _param_changed(self, name: str, value: Any):

    def get_values(self) -> dict:
        values = {}
        for param in self.parameters["base"]:
            values[param["name"]] = param["get_value"](param["widget"])

        return values

    def set_values(self, values: dict):
        for param in self.parameters["base"]:
            value = values.get(param["name"], param["default"])
            param["set_value"](param["widget"], value)
