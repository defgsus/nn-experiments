from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import torch

from ..images import LImageWidget
from ...clipig_worker import ClipigWorker
from ..task.task_config_widget import TaskConfigWidget
from ..models.preset_model import PresetModel
from .preset_widget import PresetWidget
from ..util import qimage_to_torch


class TaskWidget(QWidget):

    signal_run_task = pyqtSignal(int, dict)
    signal_stop_task = pyqtSignal(int)

    _static_id_count = 0

    def __init__(self, *args, clipig: ClipigWorker, preset_model: PresetModel, **kwargs):
        super().__init__(*args, **kwargs)
        self.clipig = clipig
        self.preset_model = preset_model
        self._source_image = None

        TaskWidget._static_id_count += 1
        self.task_id = TaskWidget._static_id_count

        self._create_widgets()

    def _create_widgets(self):
        lh = QHBoxLayout(self)

        self.image_widget = LImageWidget(self)
        lh.addWidget(self.image_widget)

        lv = QVBoxLayout()
        lh.addLayout(lv)

        self.preset_widget = PresetWidget(self, preset_model=self.preset_model)
        lv.addWidget(self.preset_widget)

        scroll = QScrollArea(self)
        lv.addWidget(scroll, stretch=100)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)

        self.config_widget = TaskConfigWidget(self, preset_model=self.preset_model)
        scroll.setWidget(self.config_widget)
        # lv.addWidget(self.config_widget)
        self.config_widget.set_values(self.preset_model.default_config()["config"])

        self.preset_widget.signal_preset_changed.connect(self.config_widget.set_values)
        self.preset_widget.signal_save_preset.connect(self._save_preset)

        self.run_button = QPushButton(self)
        lv.addWidget(self.run_button)
        self.run_button.setCheckable(True)
        self.run_button.setText(self.tr("&run"))
        self.run_button.clicked.connect(self._run_click)

        lv.addStretch(2)

    def _run_click(self):
        # self._update_run_button()

        if self.run_button.isChecked():
            config = self._get_run_config()
            self.signal_run_task.emit(self.task_id, config)
        else:
            self.signal_stop_task.emit(self.task_id)

    def _update_run_button(self):
        if self.run_button.isChecked():
            self.run_button.setText(self.tr("&stop"))
        else:
            self.run_button.setText(self.tr("&run"))

    def slot_task_event(self, event: dict):
        if "status" in event:
            status = event["status"]

            if status in ("finished", "stopped", "crashed"):
                self.run_button.setChecked(False)
            else:
                self.run_button.setChecked(True)

            self._update_run_button()

        if "message" in event:
            message = event["message"]

            if message.get("pixels") is not None:
                pixels: torch.Tensor = message["pixels"]

                self.image_widget.set_image("<clipig>", pixels)

    def _save_preset(self, name: str):
        self.preset_model.save_preset(name, self.config_widget.get_values())

    def _get_run_config(self) -> dict:
        config = self.config_widget.get_values()

        if config.get("initialize") == "input":
            if self.image_widget.image is not None:
                config["input_image"] = qimage_to_torch(self.image_widget.image)

        return config

