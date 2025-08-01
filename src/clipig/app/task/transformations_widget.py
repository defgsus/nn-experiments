from copy import deepcopy
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ...parameters import (
    get_clipig_task_parameters, get_complete_clipig_task_config, get_complete_clipig_transformation_config
)
from ..parameters import ParameterWidget
from .new_transform_dialog import NewTransformationDialog
from .transformations_preview_widget import TransformationsPreviewWidget


class TransformationsWidget(QWidget):

    signal_run_transformation_preview = pyqtSignal()

    def __init__(
            self,
            *,
            transformations: List[dict],
            default_parameters: dict,
            project: "ProjectWidget", parent: Optional[QWidget] = None,
    ):
        super().__init__(parent or project)

        from ..project import ProjectWidget
        self._project: ProjectWidget = project

        self._init_transformations = deepcopy(transformations)
        self.default_parameters = default_parameters
        self._param_widgets: List[ParameterWidget] = []

        self._create_widgets()
        self._row_changed()

    @property
    def transformations(self) -> List[dict]:
        return [
            self.list_widget.item(i).data(Qt.UserRole)
            for i in range(self.list_widget.count())
        ]

    def set_preview_images(self, images: List[QImage]):
        self.transform_preview_widget.set_preview_images(images)

    def set_running(self, running: bool):
        self.transform_preview_widget.set_running(running)

    def _create_widgets(self):
        lv0 = QVBoxLayout(self)

        lh = QHBoxLayout()
        lv0.addLayout(lh)

        lv = QVBoxLayout()
        lh.addLayout(lv)

        self.list_widget = QListWidget(self)
        self.list_widget.setFixedWidth(200)
        lv.addWidget(self.list_widget)
        self.list_widget.setDragDropMode(QAbstractItemView.DragDrop)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.model().rowsMoved.connect(self._row_changed)

        for trans in self._init_transformations:
            item = QListWidgetItem(trans["name"], parent=self.list_widget)
            item.setData(Qt.UserRole, trans)
            self.list_widget.addItem(item)

        self.list_widget.setCurrentIndex(self.list_widget.model().index(0, 0))
        self.list_widget.currentRowChanged.connect(self._row_changed)

        lh2 = QHBoxLayout()
        lv.addLayout(lh2)

        self.butt_add = QPushButton(self.tr("+"), self)
        self.butt_add.clicked.connect(self.slot_add_transform)
        lh2.addWidget(self.butt_add)
        self.butt_remove = QPushButton(self.tr("-"), self)
        self.butt_remove.clicked.connect(self.slot_remove_transform)
        lh2.addWidget(self.butt_remove)

        self.param_widget = QWidget()
        lh.addWidget(self.param_widget, stretch=10)
        self._layout = lh

        self.transform_preview_widget = TransformationsPreviewWidget(project=self._project, parent=self)
        self.transform_preview_widget.signal_run_transformation_preview.connect(
            self.signal_run_transformation_preview
        )
        lv0.addWidget(self.transform_preview_widget)

    def _row_changed(self):
        self.param_widget.deleteLater()
        self.param_widget = QWidget()
        self._layout.addWidget(self.param_widget)
        self._param_widgets = []

        lv = QVBoxLayout(self.param_widget)

        index = self.list_widget.currentRow()
        if not 0 <= index < self.list_widget.count():
            self.butt_remove.setEnabled(False)

        else:
            self.butt_remove.setEnabled(True)

            trans_values = self.transformations[index]

            for param in self.default_parameters["transformations"][trans_values["name"]]:

                widget = ParameterWidget(param, self.param_widget)
                widget.set_value(trans_values["params"][param["name"]])
                lv.addWidget(widget)
                self._param_widgets.append(widget)

                widget.signal_value_changed.connect(partial(self._value_changed, index, param["name"]))

            lv.addStretch(1)

    def _value_changed(self, index: int, param_name: str, v):
        trans = self.list_widget.item(index).data(Qt.UserRole)
        trans["params"][param_name] = v
        self.list_widget.item(index).setData(Qt.UserRole, trans)

    def slot_add_transform(self):
        trans_name = NewTransformationDialog.get_transformation_name(self)
        if trans_name:
            new_transform = get_complete_clipig_transformation_config({
                "name": trans_name,
                "params": {},
            })

            index = self.list_widget.currentRow()

            item = QListWidgetItem(trans_name, parent=self.list_widget)
            item.setData(Qt.UserRole, new_transform)
            self.list_widget.insertItem(index, item)

    def slot_remove_transform(self):
        index = self.list_widget.currentRow()
        if 0 <= index < self.list_widget.count():
            self.list_widget.takeItem(index)

