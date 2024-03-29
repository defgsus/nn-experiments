from typing import Dict, Optional

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget
from ..models import SourceModelModel
from ...parameters import get_complete_clipig_source_model_config


class SourceModelWidget(QFrame):

    def __init__(self, *args, default_parameters: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_parameters = default_parameters
        self._param_widgets: Dict[str, ParameterWidget] = {}
        self._ignore_select_widget = False

        self.setFrameStyle(QFrame.StyledPanel)
        self.setLineWidth(1)

        self._create_widgets()
        self.slot_set_source_model("pixels")

    def get_values(self) -> dict:
        return {
            "name": self.select_widget.currentText(),
            "params": {
                name: widget.get_value()
                for name, widget in self._param_widgets.items()
            }
        }

    def _create_widgets(self):
        lv = QVBoxLayout(self)

        lh = QHBoxLayout()
        lv.addLayout(lh)
        lh.addWidget(QLabel(self.tr("source model"), self))
        self.select_widget = QComboBox(self)
        self.select_widget.setModel(SourceModelModel(self))
        self.select_widget.currentTextChanged.connect(self.slot_set_source_model)
        lh.addWidget(self.select_widget)

        self.param_widget = QWidget()
        lv.addWidget(self.param_widget, stretch=10)
        self._layout = lv

    def slot_set_source_model(self, model_name: str, config: Optional[dict] = None):
        if self._ignore_select_widget:
            return

        self.param_widget.deleteLater()
        self.param_widget = QWidget()
        self._layout.addWidget(self.param_widget)
        self._param_widgets.clear()
        try:
            self._ignore_select_widget = True
            self.select_widget.setCurrentText(model_name)
        finally:
            self._ignore_select_widget = False

        lv = QVBoxLayout(self.param_widget)

        params = self.default_parameters["source_models"][model_name]
        for param in params:
            self._param_widgets[param["name"]] = widget = ParameterWidget(param, self.param_widget)
            lv.addWidget(widget)

        config = get_complete_clipig_source_model_config(config or {"name": model_name})

        for key, value in config["params"].items():
            self._param_widgets[key].set_value(value)
