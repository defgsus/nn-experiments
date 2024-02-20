import os.path
import tarfile
from copy import deepcopy
from pathlib import Path
from collections import namedtuple
from typing import Union, Optional, Tuple, Any, List

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget, ParametersWidget
from .limage import LImage, LImageLayer
from ...parameters import get_clipig_task_parameters, get_complete_clipig_transformation_config
from .limage_simple_widget import LImageSimpleWidget
from ...transformations import create_transformation


class ImageTransformDialog(QDialog):

    def __init__(
            self,
            *,
            image: QImage,
            project: "ProjectWidget",
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent or project)
        from ..project import ProjectWidget
        self._project: ProjectWidget = project
        self._limage = LImage()
        self._limage.add_layer(image=image)
        self._undo_stack: List[QImage] = []
        self.default_parameters = get_clipig_task_parameters()

        self.setWindowTitle("Image transformation")
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)

        self.setMinimumWidth(800)
        self.setMinimumHeight(800)

        self._create_widgets()

        self.set_settings(self._project.get_dialog_settings("image_transformation"))
        self.finished.connect(lambda: self._project.set_dialog_settings("image_transformation", self.get_settings()))
    @property
    def image(self) -> QImage:
        return self._limage.layers[0].image

    def get_settings(self):
        rect = self.rect()
        top_left = self.mapToGlobal(rect.topLeft())
        return {
            "window": {
                "position": [top_left.x(), top_left.y()],
                "size": [rect.width(), rect.height()],
            },
            "limage_widget": self.image_widget.get_settings(),
            "transformation": {
                "type": self.select_transform.currentText(),
                "config": self.params_widget.get_values(),
            }
        }

    def set_settings(self, settings: dict):
        if sett := settings.get("window"):
            self.setFixedSize(QSize(*sett["size"]))
        if sett := settings.get("limage_widget"):
            self.image_widget.set_settings(sett)
        if sett := settings.get("transformation"):
            self.select_transform.setCurrentText(sett["type"])
            self._set_transform(sett["type"])
            self.params_widget.set_values(sett["config"], emit=False)

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(20, 20, 20, 20)

        self.image_widget = LImageSimpleWidget(project=self._project, parent=self)
        self.image_widget.set_limage(self._limage)
        lv.addWidget(self.image_widget)

        self.select_transform = QComboBox(self)
        lv.addWidget(self.select_transform)

        for trans_name in self.default_parameters["transformations"]:
            self.select_transform.addItem(trans_name)

        self.select_transform.currentTextChanged.connect(self._set_transform)

        self.params_widget = ParametersWidget(self, exclude=["active"])
        lv.addWidget(self.params_widget)

        lh = QHBoxLayout()
        lv.addLayout(lh)
        butt = QPushButton("apply transformation")
        butt.clicked.connect(self._apply)
        lh.addWidget(butt)

        self.butt_undo = QPushButton("undo")
        self.butt_undo.clicked.connect(self._undo)
        lh.addWidget(self.butt_undo)

        lv.addSpacing(20)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    def _set_transform(self, transform_name: str):
        params = self.default_parameters["transformations"][transform_name]
        self.params_widget.set_parameters(params)

    def _apply(self):
        config = {
            "name": self.select_transform.currentText(),
            "params": self.params_widget.get_values(),
        }
        config = get_complete_clipig_transformation_config(config, self.default_parameters)
        transform = create_transformation(config["name"], config["params"])
        self._undo_stack.append(self._limage.selected_layer.image)
        self.butt_undo.setEnabled(True)
        image = self._limage.selected_layer.to_torch()
        image = transform(image)
        self._limage.selected_layer.from_torch(image)

    def _undo(self):
        if self._undo_stack:
            self._limage.selected_layer.set_image(self._undo_stack.pop())
        if not self._undo_stack:
            self.butt_undo.setEnabled(False)

    @classmethod
    def run_dialog(
            cls,
            *,
            image: QImage,
            project: "ProjectWidget",
            parent: Optional[QWidget] = None,
    ) -> Optional[QImage]:
        dialog = cls(
            image=image,
            project=project,
            parent=parent,
        )
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.image

    @classmethod
    def run_dialog_on_limage_layer(
            cls,
            *,
            limage: LImage,
            project: "ProjectWidget",
            parent: Optional[QWidget] = None,
    ) -> bool:
        dialog = cls(
            image=limage.selected_layer.image,
            project=project,
            parent=parent,
        )
        result = dialog.exec_()
        if result == QDialog.Accepted:
            limage.selected_layer.set_image(dialog.image)
            return True
        else:
            return False
