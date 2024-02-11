from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import PIL.Image
import torch

from .util import image_to_qimage
from .image_widget import ImageWidget
from ..clipig_worker import ClipigWorker
from .task_config_widget import TaskConfigWidget


class TaskWidget(QWidget):

    signal_run_task = pyqtSignal(int, dict)
    signal_stop_task = pyqtSignal(int)

    _static_id_count = 0

    def __init__(self, *args, clipig: ClipigWorker, **kwargs):
        super().__init__(*args, **kwargs)
        self.clipig = clipig
        self._source_image = None

        TaskWidget._static_id_count += 1
        self.task_id = TaskWidget._static_id_count

        self._create_widgets()

    def _create_widgets(self):
        lh = QHBoxLayout(self)

        self.image_widget = ImageWidget(self)
        lh.addWidget(self.image_widget)

        lv = QVBoxLayout()
        lh.addLayout(lv)

        self.config_widget = TaskConfigWidget(self)
        lv.addWidget(self.config_widget)

        self.run_button = QPushButton(self)
        lv.addWidget(self.run_button)
        self.run_button.setCheckable(True)
        self.run_button.setText(self.tr("run"))
        self.run_button.clicked.connect(self._run_click)

        lv.addStretch(2)

    def _run_click(self):
        # self._update_run_button()

        if self.run_button.isChecked():
            config = self.config_widget.get_values()
            self.signal_run_task.emit(self.task_id, config)
        else:
            self.signal_stop_task.emit(self.task_id)

    def _update_run_button(self):
        if self.run_button.isChecked():
            self.run_button.setText(self.tr("stop"))
        else:
            self.run_button.setText(self.tr("run"))

    def slot_task_event(self, event: dict):
        if "status" in event:
            status = event["status"]

            if status in ("finished", "stopped"):
                self.run_button.setChecked(False)
            else:
                self.run_button.setChecked(True)

            self._update_run_button()

        if "message" in event:
            message = event["message"]

            if message.get("pixels") is not None:
                pixels: torch.Tensor = message["pixels"]

                self.image_widget.set_image(pixels)
