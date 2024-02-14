import os.path
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Tuple, Any

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..parameter_widget import ParameterWidget
from .limage import LImage, LImageLayer


class NewLayerDialog(QDialog):

    def __init__(
            self,
            limage: LImage,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.limage = limage
        self.added_layer = None
        self.setWindowTitle("Add layer")

        self._create_widgets()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(20, 20, 20, 20)

        self.name_widget = ParameterWidget({
            "type": "str",
            "name": "name",
            "default": f"layer #{len(self.limage.layers) + 1}",
        }, self)
        lv.addWidget(self.name_widget)

        size = self.limage.size()
        size = (size.width(), size.height())
        if size == (0, 0):
            size = (128, 128)

        self.size_widget = ParameterWidget({
            "type": "int2",
            "name": "width and height",
            "default": size,
            "min": (1, 1),
        }, self)
        lv.addWidget(self.size_widget)

        lv.addSpacing(20)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self._accept_clicked)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    def _accept_clicked(self):
        image = QImage(QSize(*self.size_widget.get_value()), QImage.Format.Format_ARGB32)

        self.added_layer = self.limage.add_layer(
            name=self.name_widget.get_value(),
            image=image,
        )
        self.accept()

    @classmethod
    def run_new_layer_dialog(
            cls,
            limage: LImage,
            parent: Optional[QWidget] = None,
    ) -> Optional[LImageLayer]:
        dialog = cls(
            limage=limage,
            parent=parent,
        )
        result = dialog.exec_()
        return dialog.added_layer
