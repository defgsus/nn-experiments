import os.path
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Tuple, Any

import yaml
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget


class ResizeDialog(QDialog):

    def __init__(
            self,
            size: Tuple[int, int],
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._size = size
        if title is None:
            title = self.tr("Resize")

        self.setWindowTitle(title)

        self._create_widgets()

    def get_size(self):
        return tuple(w.get_value() for w in self.size_widgets)

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(20, 20, 20, 20)

        self.size_widgets = [
            ParameterWidget({
                "name": name,
                "type": "int",
                "default": self._size,
                "min": 1,
                "max": 2*16,
            }, parent=self)
            for name in ("width", "height")
        ]
        self.size_widgets[0].signal_value_changed.connect(self._width_changed)
        self.size_widgets[1].signal_value_changed.connect(self._height_changed)
        for w in self.size_widgets:
            lv.addWidget(w)

        lv.addSpacing(20)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    def _width_changed(self, v):
        pass

    def _height_changed(self, v):
        pass

    @classmethod
    def new_size_dialog(
            cls,
            size: Tuple[int, int],
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ) -> Tuple[bool, Any]:
        dialog = cls(
            size=size,
            title=title,
            parent=parent,
        )
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return True, dialog.get_size()

        return False, None
