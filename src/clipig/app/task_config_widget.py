import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import yaml

from .preset_model import PresetModel
from ..task_parameters import get_task_parameters, get_complete_task_config, get_complete_transformation_config
from .parameter_widget import ParameterWidget
from .new_transform_dialog import NewTransformationDialog


class TaskConfigWidget(QWidget):

    def __init__(self, *args, preset_model: PresetModel, **kwargs):
        super().__init__(*args, **kwargs)

        self.preset_model = preset_model
        self._base_widgets: List[dict] = []
        self._target_widgets: List[dict] = []

        self.default_parameters = get_task_parameters()

        self._create_widgets()
        self.set_values({})

    def _create_widgets(self):
        lv = QVBoxLayout(self)

        self._base_widgets.clear()
        for param in self.default_parameters["base"]:
            param = deepcopy(param)
            self._base_widgets.append(param)

            widget = param["widget"] = ParameterWidget(param, self)
            lv.addWidget(widget)

        self.target_tab_widget = QTabWidget(self)
        lv.addWidget(self.target_tab_widget)

    def get_values(self) -> dict:
        values = {}
        for param in self._base_widgets:
            values[param["name"]] = param["widget"].get_value()

        values["targets"] = []
        for target_params in self._target_widgets:
            values["targets"].append({})

            for param in target_params["params"]:
                values["targets"][-1][param["name"]] = param["widget"].get_value()

            values["targets"][-1]["transformations"] = target_params["transformation_widget"].transformations

        return deepcopy(values)

    def set_values(self, values: dict):
        values = get_complete_task_config(values)

        for param in self._base_widgets:
            value = values[param["name"]]
            param["widget"].set_value(value)

        # create a tab for each target and all of it's widgets for value readout

        self.target_tab_widget.clear()
        self._target_widgets.clear()

        for target_values in values["targets"]:
            tab = QWidget()
            lv = QVBoxLayout(tab)
            self.target_tab_widget.addTab(tab, self.tr("target #{i}").format(i=self.target_tab_widget.count() + 1))

            target_params = {
                "params": [],
                "transformation_widget": None,
            }
            self._target_widgets.append(target_params)

            for param in self.default_parameters["target"]:
                param = deepcopy(param)
                target_params["params"].append(param)

                param["widget"] = widget = ParameterWidget(param, tab)
                widget.set_value(target_values[param["name"]])
                lv.addWidget(widget)

            lv.addWidget(QLabel(self.tr("transformations"), tab))
            target_params["transformation_widget"] = trans_widget = TransformationsWidget(
                tab, transformations=target_values["transformations"], default_parameters=self.default_parameters,
            )
            lv.addWidget(trans_widget)

        butt = QPushButton(self.tr("new target"), self)
        butt.clicked.connect(self.slot_new_target)
        self.target_tab_widget.addTab(butt, "+")

    def slot_new_target(self):
        default_config = self.preset_model.default_config()["config"]
        config = self.get_values()
        config["targets"].append(
            deepcopy(default_config["targets"][0])
        )
        self.set_values(config)
        self.target_tab_widget.setCurrentIndex(self.target_tab_widget.count() - 2)


class TransformationsWidget(QWidget):

    def __init__(self, *args, transformations: List[dict], default_parameters: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_transformations = deepcopy(transformations)
        self.default_parameters = default_parameters
        self._param_widgets: List[ParameterWidget] = []

        self._create_widgets()
        self._row_changed(0)

    @property
    def transformations(self) -> List[dict]:
        return [
            self.list_widget.item(i).data(Qt.UserRole)
            for i in range(self.list_widget.count())
        ]

    def _create_widgets(self):
        lh = QHBoxLayout(self)

        lv = QVBoxLayout()
        lh.addLayout(lv)

        self.list_widget = QListWidget(self)
        self.list_widget.setFixedWidth(200)
        lv.addWidget(self.list_widget)
        self.list_widget.setDragDropMode(QAbstractItemView.DragDrop)

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

    def _row_changed(self, index: int):
        self.param_widget.deleteLater()
        self.param_widget = QWidget()
        self._layout.addWidget(self.param_widget)
        self._param_widgets = []

        lv = QVBoxLayout(self.param_widget)

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
            new_transform = get_complete_transformation_config({
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
