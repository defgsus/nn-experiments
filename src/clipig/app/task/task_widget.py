import os.path
import tarfile
from pathlib import Path
from typing import Union

import yaml
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import torch

from ..images import LImageWidget
from ...clipig_task import ClipigTask
from ...clipig_worker import ClipigWorker
from ..task.task_config_widget import TaskConfigWidget
from ..models.preset_model import PresetModel
from .preset_widget import PresetWidget
from ..util import qimage_to_torch, image_to_qimage
from ..images import LImage
from src.util.files import Filestream


class TaskWidget(QWidget):

    signal_changed = pyqtSignal()

    signal_run_task = pyqtSignal(int, dict)
    signal_stop_task = pyqtSignal(int)
    signal_new_task_with_image = pyqtSignal(LImage)
    signal_copy_task = pyqtSignal()

    _static_id_count = 0

    def __init__(self, *args, project: "ProjectWidget", **kwargs):
        super().__init__(*args, **kwargs)
        from ..project import ProjectWidget

        self._project: ProjectWidget = project
        self._source_image = None

        TaskWidget._static_id_count += 1
        self.task_id = TaskWidget._static_id_count

        self._create_widgets()

    @property
    def clipig(self):
        return self._project.clipig

    @property
    def preset_model(self):
        return self._project.preset_model

    def _create_widgets(self):
        lh = QHBoxLayout(self)

        self.image_widget = LImageWidget(project=self._project, parent=self)
        lh.addWidget(self.image_widget)
        self.image_widget.signal_new_task_with_image.connect(self.signal_new_task_with_image)
        self.image_widget.signal_changed.connect(self.set_changed)

        lv = QVBoxLayout()
        lh.addLayout(lv)

        self.preset_widget = PresetWidget(self, preset_model=self.preset_model)
        lv.addWidget(self.preset_widget)

        scroll = QScrollArea(self)
        lv.addWidget(scroll, stretch=100)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)

        self.config_widget = TaskConfigWidget(project=self.preset_model, parent=self)
        scroll.setWidget(self.config_widget)
        # lv.addWidget(self.config_widget)
        self.config_widget.set_values(self.preset_model.default_config()["config"])
        self.config_widget.signal_run_transformation_preview.connect(self._run_transformation_preview)
        self.preset_widget.signal_preset_changed.connect(self.config_widget.set_values)
        self.preset_widget.signal_save_preset.connect(self._save_preset)

        self.run_button = QPushButton(self)
        lv.addWidget(self.run_button)
        self.run_button.setCheckable(True)
        self.run_button.setText(self.tr("&run"))
        self.run_button.clicked.connect(self._run_click)

        lv.addStretch(2)

    def add_menu_actions(self, menu: QMenu):
        menu.addAction(self.tr("Clone Task"), self.signal_copy_task)

    def set_changed(self):
        self.signal_changed.emit()

    @property
    def limage(self) -> LImage:
        return self.image_widget.limage

    def set_limage(self, image: LImage):
        self.image_widget.set_limage(image)
        self.set_changed()

    def get_task_config(self) -> dict:
        return self.config_widget.get_values()

    def set_task_config(self, config: dict):
        self.config_widget.set_values(config)

    def get_settings(self) -> dict:
        return {}

    def set_settings(self, settings: dict):
        self.set_changed()

    def set_running(self, running: bool):
        self.config_widget.set_running(running)

    def _run_click(self):
        # self._update_run_button()

        if self.run_button.isChecked():
            config = self._get_run_config()
            self.signal_run_task.emit(self.task_id, config)
        else:
            self.signal_stop_task.emit(self.task_id)

    def _run_transformation_preview(self):
        config = self._get_run_config()
        config["task_type"] = ClipigTask.TaskType.T_TRANSFORMATION_PREVIEW
        self.signal_run_task.emit(self.task_id, config)

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
                self.set_running(False)
            else:
                self.run_button.setChecked(True)
                self.set_running(True)

            self._update_run_button()

        if "message" in event:
            message = event["message"]

            if message.get("pixels") is not None:
                pixels: torch.Tensor = message["pixels"]

                self.image_widget.set_image("<clipig>", pixels)
                self.set_changed()

            if message.get("transformation_preview") is not None:
                target_index = message["transformation_preview"]["target_index"]
                images = [
                    image_to_qimage(pixels)
                    for pixels in message["transformation_preview"]["pixels"]
                ]
                self.config_widget.set_transformation_preview(target_index, images)

    def _save_preset(self, name: str):
        self.preset_model.save_preset(name, self.config_widget.get_values())

    def _get_run_config(self) -> dict:
        config = self.config_widget.get_values()

        if config.get("initialize") == "input":
            if self.image_widget.limage is not None and not self.image_widget.limage.is_empty():
                config["input_image"] = self.image_widget.limage.to_torch()

        return config

    def save_to_filestream(self, filestream: Filestream, directory: Union[str, Path]):
        directory = Path(directory)

        # -- write config --

        config_data = {
            "task_settings": self.get_settings(),
            "image_settings": self.image_widget.get_settings(),
            "task_config": self.config_widget.get_values(),
        }
        if not self.image_widget.limage.is_empty():
            config_data["limage_config_filename"] = str(directory / "limage" / "config.yaml")

        filestream.write_yaml(directory / "config.yaml", config_data)

        # -- write LImage --

        if config_data.get("limage_config_filename"):
            self.image_widget.limage.save_to_filestream(
                filestream,
                Path(config_data["limage_config_filename"]).parent,
            )

    def load_from_filestream(self, filestream: Filestream, config_filename: Union[str, Path]):
        config_data = filestream.read_yaml(config_filename)

        limage = LImage()
        if config_data.get("limage_config_filename"):
            limage.load_from_filestream(filestream, config_data["limage_config_filename"])

        self.set_limage(limage)

        self.image_widget.set_settings(config_data["image_settings"])
        self.set_settings(config_data["task_settings"])
        self.config_widget.set_values(config_data["task_config"])

        self.set_changed()
