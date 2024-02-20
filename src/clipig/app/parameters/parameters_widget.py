import json
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .parameter_widget import ParameterWidget


class ParametersWidget(QWidget):

    signal_value_changed = pyqtSignal(str, QVariant)
    signal_values_changed = pyqtSignal(dict)

    def __init__(
            self,
            parent: Optional[QWidget],
            parameters: Optional[List[dict]] = None,
            values: Optional[Dict[str, Any]] = None,
            exclude: Optional[Iterable[str]] = None,
    ):
        super().__init__(parent)

        self._parameters: List[dict] = parameters or []
        self._widget_map: Dict[str, ParameterWidget] = {}
        self._exclude = set(exclude) if exclude is not None else []
        self._backup_values: Dict[str, Any] = {}

        self._create_widgets()

        if values:
            self.set_values(values, emit=False)

    @property
    def parameters(self) -> List[dict]:
        return self._parameters

    def get_values(self) -> Dict[str, Any]:
        return {
            key: widget.get_value()
            for key, widget in self._widget_map.items()
            if key not in self._exclude
        }

    def set_value(self, name: str, value: Any, emit: bool):
        if name not in self._exclude:
            if widget := self._widget_map.get(name):
                widget.set_value(value, emit=emit)
                self._backup_values[name] = value

    def set_values(self, values: Dict[str, Any], emit: bool):
        for key, widget in self._widget_map.items():
            if key in values and key not in self._exclude:
                widget.set_value(values[key], emit=emit)
                self._backup_values[key] = values[key]

    def set_parameters(self, parameters: List[dict], values: Optional[Dict[str, Any]] = None, emit: bool = False):
        self._parameters = deepcopy(parameters)
        self._create_param_widgets()

        if values is None:
            values = self._backup_values
        else:
            values = {
                **self._backup_values,
                **values,
            }

        self.set_values(values, emit=emit)

    def _create_widgets(self):
        self._layout = QVBoxLayout(self)
        self._widget_container = QWidget()
        self._layout.addWidget(self._widget_container)

        self._create_param_widgets()

    def _create_param_widgets(self):
        self._widget_container.close()
        self._widget_container.deleteLater()
        self._widget_container = QWidget()
        self._layout.addWidget(self._widget_container)

        self._widget_map.clear()
        lv = QVBoxLayout(self._widget_container)
        for param in self._parameters:
            if param["name"] not in self._exclude:
                self._widget_map[param["name"]] = widget = ParameterWidget(param, parent=self._widget_container)
                lv.addWidget(widget)
                widget.signal_value_changed.connect(partial(self._value_changed, param["name"]))

    def _value_changed(self, name: str, v: Any):
        self._backup_values[name] = v
        self.signal_value_changed.emit(name, QVariant(v))
        self.signal_values_changed.emit(self.get_values())
