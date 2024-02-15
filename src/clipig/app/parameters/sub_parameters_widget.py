import json
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .parameter_widget import ParameterWidget
from .parameters_widget import ParametersWidget


class SubParametersWidget(QWidget):

    signal_values_changed = pyqtSignal(dict)
    signal_select_value_changed = pyqtSignal(str)

    def __init__(
            self,
            parent: Optional[QWidget],
            select_parameter: dict,
            sub_parameters: Dict[str, List[dict]],
            values: Optional[Dict[str, Any]] = None,
    ):
        """
        A selection parameter with a set of sub-parameters per selected value.

        :param parent: QWidget
        :param select_parameter: dict, a "type": "select" parameter that chooses the sub-parameter set
        :param sub_parameters: dict of "<selected value>" -> list of parameters
        :param values: optional values for everything
        """
        super().__init__(parent)

        self._select_parameter = select_parameter
        self._sub_parameters = sub_parameters

        self._create_widgets()

        if values:
            self.set_values(values, emit=False)

    @property
    def select_parameter(self) -> dict:
        return self._select_parameter

    @property
    def sub_parameters(self) -> Dict[str, List[dict]]:
        return self._sub_parameters

    @property
    def selected_value(self):
        return self._select_widget.get_value()

    def get_values(self) -> Dict[str, Any]:
        return {
            self._select_parameter["name"]: self._select_widget.get_value(),
            **self._params_wiget.get_values(),
        }

    def set_selected_value(self, value: str, emit: bool):
        self._select_widget.set_value(value, emit=False)
        self._params_widget.set_parameters(self._sub_parameters.get(value) or [], emit=False)
        if emit:
            self.signal_select_value_changed.emit(value)
            self.signal_values_changed.emit(self.get_values())

    def set_value(self, name: str, value: Any, emit: bool):
        if name == self._select_parameter["name"]:
            self.set_selected_value(value, emit=emit)
            return

        self._params_widget.set_value(name, value, emit=emit)

    def set_values(self, values: Dict[str, Any], emit: bool):

        emit_select = False
        if self._select_parameter["name"] in values:
            emit_select = True
            select_value = values[self._select_parameter["name"]]
            self.set_selected_value(select_value, emit=False)

        self._params_widget.set_values(values, emit=False)

        if emit:
            if emit_select:
                self.signal_select_value_changed.emit(self._select_widget.get_value())
            self.signal_values_changed.emit(self.get_values())

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self._select_widget = ParameterWidget(self._select_parameter, self)
        self._select_widget.signal_value_changed.connect(self._select_changed)
        lv.addWidget(self._select_widget)

        lh = QHBoxLayout()
        lh.setContentsMargins(20, 0, 0, 0)
        lv.addLayout(lh)

        self._params_widget = ParametersWidget(self, parameters=self._sub_parameters.get(self.selected_value) or [])
        self._params_widget.signal_value_changed.connect(self._value_changed)
        lh.addWidget(self._params_widget)

    def _select_changed(self, v):
        self._params_widget.set_parameters(self._sub_parameters.get(v) or [], emit=False)
        self.signal_values_changed.emit(self.get_values())

    def _value_changed(self, name: str, v: Any):
        self.signal_values_changed.emit(self.get_values())
