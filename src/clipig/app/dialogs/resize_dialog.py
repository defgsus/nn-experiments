import os.path
import tarfile
from copy import deepcopy
from pathlib import Path
from collections import namedtuple
from typing import Union, Optional, Tuple, Any

import yaml
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget
from ...transformations.base import INTERPOLATION_PARAMETER


ResizeResult = namedtuple("ResizeResult", ["size", "transform_mode", "aspect_mode"])


class ResizeDialog(QDialog):

    def __init__(
            self,
            size: QSize,
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._size = (size.width(), size.height())
        if title is None:
            title = self.tr("Resize")

        self.setWindowTitle(title)
        self.setMinimumWidth(500)

        self._create_widgets()

    def get_result(self) -> ResizeResult:
        return ResizeResult(
            tuple(w.get_value() for w in self.size_widgets),
            {
                "none": Qt.TransformationMode.FastTransformation,
                "smooth": Qt.TransformationMode.SmoothTransformation,
            }[self.interpolation_widget.get_value()],
            {
                "ignore": Qt.AspectRatioMode.IgnoreAspectRatio,
                "shrink to fit": Qt.AspectRatioMode.KeepAspectRatio,
                "expand to fit": Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            }[self.aspect_widget.get_value()]
        )

    def get_interpolation(self):
        return self.interpolation_widget.get_value()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(20, 20, 20, 20)

        self.size_widgets = [
            ParameterWidget({
                "name": name,
                "type": "int",
                "default": self._size[i],
                "min": 1,
                "max": 2**16,
            }, parent=self)
            for i, name in enumerate(("width", "height"))
        ]
        self.size_widgets[0].signal_value_changed.connect(self._width_changed)
        self.size_widgets[1].signal_value_changed.connect(self._height_changed)
        for w in self.size_widgets:
            lv.addWidget(w)

        self.interpolation_widget = ParameterWidget({
            "name": "interpolation",
            "type": "select",
            "default": "smooth",
            "choices": ["none", "smooth"],
        }, parent=self)
        lv.addWidget(self.interpolation_widget)

        self.aspect_widget = ParameterWidget({
            "name": "aspect ratio",
            "type": "select",
            "default": "ignore",
            "choices": ["ignore", "shrink to fit", "expand to fit"],
        }, parent=self)
        lv.addWidget(self.aspect_widget)

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
    def run_dialog(
            cls,
            size: QSize,
            title: Optional[str] = None,
            parent: Optional[QWidget] = None,
    ) -> Optional[ResizeResult]:
        dialog = cls(
            size=size,
            title=title,
            parent=parent,
        )
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.get_result()

