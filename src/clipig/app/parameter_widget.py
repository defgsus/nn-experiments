import json
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ParameterWidget(QWidget):

    signal_value_changed = pyqtSignal(QVariant)

    _RE_INT_N = re.compile(r"^int(\d+)$")
    _RE_FLOAT_N = re.compile(r"^float(\d+)$")

    def __init__(self, parameter: dict, parent: QWidget):
        super().__init__(parent)

        self.parameter = parameter
        self._widget = None
        self._widgets = []

        self.setContentsMargins(0, 0, 0, 0)
        self._create_widgets()

    @property
    def parameter_type(self) -> str:
        return self.parameter["type"]

    def get_value(self):
        if self.parameter_type == "bool":
            return self._widget.checked()

        elif self.parameter_type == "str":
            return self._widget.text()

        elif self.parameter_type in ("int", "float"):
            return self._widget.value()

        elif self.parameter_type == "select":
            return self._widget.currentText()

        elif (self._RE_INT_N.match(self.parameter_type) or self._RE_FLOAT_N.match(self.parameter_type)):
            return tuple(
                w.value() for w in self._widgets
            )

    def set_value(self, value):
        if self.parameter_type == "bool":
            self._widget.setChecked(value)

        elif self.parameter_type == "str":
            self._widget.setText(value)

        elif self.parameter_type in ("int", "float"):
            self._widget.setValue(value)

        elif self.parameter_type == "select":
            self._widget.setCurrentText(value)

        elif self._RE_INT_N.match(self.parameter_type) or self._RE_FLOAT_N.match(self.parameter_type):
            for w, v in zip(self._widgets, value):
                w.setValue(v)

    def _create_widgets(self):
        if self.parameter_type == "bool":
            self._widget = QCheckBox(self)
            self._widget.stateChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "str":
            self._widget = QLineEdit(self)
            self._widget.textChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "int":
            self._widget = QSpinBox(self)
            self._widget.setLocale(QLocale("en"))
            self._widget.setRange(self.parameter.get("min", 0), self.parameter.get("max", 2 ** 24))
            self._widget.setSingleStep(self.parameter.get("step", 1))
            self._widget.valueChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "float":
            self._widget = QDoubleSpinBox(self)
            self._widget.setLocale(QLocale("en"))
            self._widget.setRange(self.parameter.get("min", 0), self.parameter.get("max", 2 ** 24))
            self._widget.setSingleStep(self.parameter.get("step", 0.1))
            self._widget.setDecimals(self.parameter.get("decimals", 10))
            self._widget.valueChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif self.parameter_type == "select":
            self._widget = QComboBox(self)
            for choice in self.parameter["choices"]:
                self._widget.addItem(choice)
            self._widget.currentTextChanged.connect(lambda v: self.signal_value_changed.emit(QVariant(v)))

        elif match := (self._RE_INT_N.match(self.parameter_type) or self._RE_FLOAT_N.match(self.parameter_type)):
            count = int(match.groups()[0])
            self._widget = self._create_multi_spinbox(count)

        else:
            raise ValueError(f"Unhandled type '{self.parameter_type} in parameter: {self.parameter}")

        lh = QHBoxLayout(self)
        lh.setContentsMargins(0, 0, 0, 0)
        lh.addWidget(QLabel(self.parameter["name"], self))
        lh.addWidget(self._widget)

    def _create_multi_spinbox(self, count: int):
        is_float = self.parameter_type.startswith("float")

        widget = QWidget(self)
        widget.setContentsMargins(0, 0, 0, 0)
        lh = QHBoxLayout(widget)
        lh.setContentsMargins(0, 0, 0, 0)

        min_values = self.parameter.get("min", [0] * count)
        max_values = self.parameter.get("max", [2**24] * count)
        step_values = self.parameter.get("step", [.1 if is_float else 1] * count)
        decimals_values = self.parameter.get("decimals", [10] * count)

        spinboxes = []
        for i in range(count):
            spinbox = QDoubleSpinBox(self) if is_float else QSpinBox(self)
            spinbox.setLocale(QLocale("en"))
            spinboxes.append(spinbox)
            lh.addWidget(spinbox)

            spinbox.setRange(min_values[i], max_values[i])
            spinbox.setSingleStep(step_values[i])
            if is_float:
                spinbox.setDecimals(decimals_values[i])

        def _on_change(v):
            values = [s.value() for s in spinboxes]
            self.signal_value_changed.emit(QVariant(values))

        for spinbox in spinboxes:
            spinbox.valueChanged.connect(_on_change)

        self._widgets = spinboxes

        return widget
