import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ParameterWidget(QWidget):

    signal_value_changed = pyqtSignal(QVariant)

    def __init__(self, parameter: dict, parent: QWidget):
        super().__init__(parent)

        self.parameter = parameter

        self._create_widgets()

    @property
    def parameter_type(self) -> str:
        return self.parameter["type"]

    def get_value(self):
        if self.parameter_type == "str":
            return self.widget.text()

        elif self.parameter_type in ("int", "float"):
            return self.widget.value()

        elif self.parameter_type == "select":
            return self.widget.currentText()

    def set_value(self, value):
        if self.parameter_type == "str":
            self.widget.setText(value)

        elif self.parameter_type in ("int", "float"):
            self.widget.setValue(value)

        elif self.parameter_type == "select":
            self.widget.setCurrentText(value)

    def _create_widgets(self):
        if self.parameter_type == "str":
            self.widget = QLineEdit(self)
            self.widget.textChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "int":
            self.widget = QSpinBox(self)
            self.widget.setRange(self.parameter.get("min", 0), self.parameter.get("max", 2**24))
            self.widget.valueChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "float":
            self.widget = QDoubleSpinBox(self)
            self.widget.setRange(self.parameter.get("min", 0), self.parameter.get("max", 2**24))
            self.widget.setSingleStep(self.parameter.get("step", 0.1))
            self.widget.valueChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "select":
            self.widget = QComboBox(self)
            for choice in self.parameter["choices"]:
                self.widget.addItem(choice)
            self.widget.currentTextChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        else:
            raise ValueError(f"Unhandled type '{self.parameter_type} in parameter: {self.parameter}")

        lh = QHBoxLayout(self)
        lh.addWidget(QLabel(self.parameter["name"], self))
        lh.addWidget(self.widget)
