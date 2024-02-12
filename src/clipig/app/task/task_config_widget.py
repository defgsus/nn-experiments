from copy import deepcopy
from typing import List

from PyQt5.QtWidgets import *

from ..models.preset_model import PresetModel
from ...task_parameters import get_task_parameters, get_complete_task_config
from ..parameter_widget import ParameterWidget
from .transformations_widget import TransformationsWidget
from .source_model_widget import SourceModelWidget


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

        self.source_model_widget = SourceModelWidget(
            self, default_parameters=self.default_parameters,
        )
        lv.addWidget(self.source_model_widget)

        self.target_tab_widget = QTabWidget(self)
        lv.addWidget(self.target_tab_widget)

    def get_values(self) -> dict:
        values = {}
        for param in self._base_widgets:
            values[param["name"]] = param["widget"].get_value()

        values["source_model"] = self.source_model_widget.get_values()

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
