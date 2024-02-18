from functools import partial
from copy import deepcopy
from typing import List, Optional, Dict

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..models.preset_model import PresetModel
from ...parameters import get_clipig_task_parameters, get_complete_clipig_task_config
from ..parameters import ParameterWidget, ParametersWidget, SubParametersWidget
from .transformations_widget import TransformationsWidget
from .source_model_widget import SourceModelWidget


class TargetFeaturesWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_parameters = get_clipig_task_parameters()
        self._feature_widgets: List[FeatureWidget] = []

        self._create_widgets()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        self._layout = lv

        lh = QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(lh)

        butt = QPushButton(self.tr("add"), self)
        butt.setToolTip(self.tr("Add a new target feature"))
        butt.clicked.connect(self.add_feature)
        lh.addWidget(butt)

        lh.addWidget(QLabel(self.tr("target features"), self), stretch=100)

    def get_values(self) -> List[dict]:
        return [
            widget.get_values()
            for widget in self._feature_widgets
        ]

    def set_values(self, values: List[dict]):
        for widget in self._feature_widgets:
            self._layout.removeWidget(widget)
            widget.deleteLater()

        self._feature_widgets.clear()
        for idx, values in enumerate(values):
            widget = FeatureWidget(self, self.default_parameters["target_feature"])
            widget.signal_remove.connect(partial(self.remove_feature, idx))
            widget.set_values(values)
            self._feature_widgets.append(widget)
            self._layout.addWidget(widget)

    def remove_feature(self, index: int):
        features = self.get_values()
        if 0 <= index < len(features):
            features.pop(index)
            self.set_values(features)

    def add_feature(self):
        features = self.get_values()
        features.append({
            param["name"]: param["default"]
            for param in self.default_parameters["target_feature"]
        })
        self.set_values(features)


class FeatureWidget(QWidget):

    signal_remove = pyqtSignal()

    def __init__(self, parent: QWidget, parameters: List[dict]):
        super().__init__(parent)

        self._parameters = parameters
        self._widgets: Dict[str, ParameterWidget] = {}

        self._create_widgets()

    def get_values(self) -> dict:
        return {
            key: widget.get_value()
            for key, widget in self._widgets.items()
        }

    def set_values(self, values: dict, emit: bool = False):
        for key, value in values.items():
            if key in self._widgets:
                self._widgets[key].set_value(value, emit=emit)

    def _create_widgets(self):
        lh = QHBoxLayout(self)
        lh.setContentsMargins(0, 0, 0, 0)

        param = next(iter(filter(lambda p: p["name"] == "weight", self._parameters)))
        self._widgets["weight"] = widget = ParameterWidget(param, parent=self, show_label=False)
        lh.addWidget(widget)

        param = next(iter(filter(lambda p: p["name"] == "text", self._parameters)))
        self._widgets["text"] = widget = ParameterWidget(param, parent=self, show_label=False)
        lh.addWidget(widget)

        butt = QPushButton("X", self)
        butt.setToolTip(self.tr("Remove feature"))
        butt.clicked.connect(self.signal_remove)
        lh.addWidget(butt)