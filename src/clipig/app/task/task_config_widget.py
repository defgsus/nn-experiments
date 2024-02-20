from copy import deepcopy
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..models.preset_model import PresetModel
from ...parameters import get_clipig_task_parameters, get_complete_clipig_task_config
from ..parameters import ParameterWidget, ParametersWidget, SubParametersWidget
from .transformations_widget import TransformationsWidget
from .source_model_widget import SourceModelWidget
from .target_features_widget import TargetFeaturesWidget


class TaskConfigWidget(QWidget):

    signal_run_transformation_preview = pyqtSignal()

    def __init__(self, *, project: "ProjectWidget", parent: Optional[QWidget] = None):
        super().__init__(parent or project)

        from ..project import ProjectWidget
        self._project: ProjectWidget = project

        self.base_params_widget: Optional[ParametersWidget] = None
        self._target_widgets: List[dict] = []

        self.default_parameters = get_clipig_task_parameters()

        self._create_widgets()
        self.set_values({})

    @property
    def preset_model(self):
        return self._project.preset_model

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self.base_params_widget = ParametersWidget(
            self, self.default_parameters["base"]
        )
        lv.addWidget(self.base_params_widget)

        self.source_model_widget = SourceModelWidget(
            self, default_parameters=self.default_parameters,
        )
        lv.addWidget(self.source_model_widget)

        self.target_tab_widget = QTabWidget(self)
        lv.addWidget(self.target_tab_widget)

    def get_values(self) -> dict:
        values = self.base_params_widget.get_values()

        values["source_model"] = self.source_model_widget.get_values()

        values["targets"] = []
        for target_widgets in self._target_widgets:

            target_values = {
                **target_widgets["params_widget"].get_values(),
                "target_features": target_widgets["features_widget"].get_values(),
                "optimizer": target_widgets["optimizer_params_widget"].get_values(),
                "transformations": target_widgets["transformation_widget"].transformations,
            }
            values["targets"].append(target_values)

        return deepcopy(values)

    def set_values(self, values: dict, emit: bool = False):
        values = get_complete_clipig_task_config(values)

        self.base_params_widget.set_values(values, emit=emit)

        self.source_model_widget.slot_set_source_model(values["source_model"]["name"], values["source_model"])

        # create a tab for each target and all of it's widgets for value readout

        self.target_tab_widget.clear()
        self._target_widgets.clear()

        for target_values in values["targets"]:
            tab = QWidget()
            lv = QVBoxLayout(tab)
            self.target_tab_widget.addTab(tab, self.tr("target #{i}").format(i=self.target_tab_widget.count() + 1))

            optimizer_param = list(filter(lambda p: p["name"] == "optimizer", self.default_parameters["target"]))[0]
            target_widgets = {
                "params_widget": ParametersWidget(
                    parent=tab,
                    parameters=self.default_parameters["target"],
                    exclude=["optimizer"],
                    values=target_values,
                ),
                "features_widget": TargetFeaturesWidget(
                    parent=tab,
                ),
                "optimizer_params_widget": SubParametersWidget(
                    parent=tab,
                    select_parameter=optimizer_param,
                    sub_parameters=self.default_parameters["optimizers"],
                    values=target_values["optimizer"],
                ),
                "transformation_widget": None,
            }
            self._target_widgets.append(target_widgets)
            lv.addWidget(target_widgets["params_widget"])
            lv.addWidget(target_widgets["features_widget"])
            lv.addWidget(target_widgets["optimizer_params_widget"])

            target_widgets["features_widget"].set_values(target_values["target_features"])

            lv.addWidget(QLabel(self.tr("transformations"), tab))
            target_widgets["transformation_widget"] = trans_widget = TransformationsWidget(
                transformations=target_values["transformations"], default_parameters=self.default_parameters,
                parent=tab, project=self._project,
            )
            trans_widget.signal_run_transformation_preview.connect(self.signal_run_transformation_preview)
            lv.addWidget(trans_widget)

        butt = QPushButton(self.tr("new target"), self)
        butt.clicked.connect(self.slot_new_target)
        self.target_tab_widget.addTab(butt, "+")

    def set_running(self, running: bool):
        for w in self._target_widgets:
            w["transformation_widget"].set_running(running)

    def slot_new_target(self):
        default_config = self.preset_model.default_config()["config"]
        config = self.get_values()
        config["targets"].append(
            deepcopy(default_config["targets"][0])
        )
        self.set_values(config)
        self.target_tab_widget.setCurrentIndex(self.target_tab_widget.count() - 2)

    def set_transformation_preview(self, target_index: int, images: List[QImage]):
        if 0 <= target_index < len(self._target_widgets):
            self._target_widgets[target_index]["transformation_widget"].set_preview_images(images)
