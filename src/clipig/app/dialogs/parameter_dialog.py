import os.path
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Tuple, Any

import yaml
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget


class ParameterDialog(QDialog):

    def __init__(
            self,
            parameter: dict,
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._parameter = deepcopy(parameter)
        if not self._parameter.get("name"):
            self._parameter["name"] = f"{self._parameter['type']} value"

        if title is None:
            title = f"choose {self._parameter['name']}"

        self.setWindowTitle(title)

        self._create_widgets()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(20, 20, 20, 20)

        self.param_widget = ParameterWidget(self._parameter, self)
        lv.addWidget(self.param_widget)

        lv.addSpacing(20)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    @classmethod
    def get_parameter_value(
            cls,
            parameter: dict,
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ) -> Tuple[bool, Any]:
        dialog = cls(
            parameter=parameter,
            title=title,
            parent=parent,
        )
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return True, dialog.param_widget.get_value()

        return False, None

    @classmethod
    def get_string_value(
            cls,
            default: str = "",
            name: Optional[str] = None,
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ) -> Tuple[bool, Any]:
        parameter = {
            "type": "str",
            "name": name,
            "default": default,
        }
        return cls.get_parameter_value(parameter, title=title, parent=parent)
