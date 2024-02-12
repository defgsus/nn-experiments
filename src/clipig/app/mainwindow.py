import json
import os
from pathlib import Path
from functools import partial
from copy import deepcopy
from typing import List, Dict, Hashable

import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..clipig_worker import ClipigWorker
from .image_widget import ImageWidget
from .task_widget import TaskWidget
from .preset_model import PresetModel


class MainWindow(QMainWindow):

    def __init__(self, clipig: ClipigWorker):
        super().__init__()

        self.clipig = clipig
        self._task_map: Dict[Hashable, dict] = {}
        self._refresh_msec = 300
        self.preset_model = PresetModel(self)

        self.setWindowTitle(self.tr("CLIPIG"))
        # self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)

        self._create_main_menu()
        self._create_widgets()
        QTimer.singleShot(self._refresh_msec, self._slot_idle)

    def _create_main_menu(self):
        menu = self.menuBar().addMenu(self.tr("&File"))

        menu.addAction(self.tr("New &Task"), self.slot_new_task)

        menu.addAction(self.tr("E&xit"), self.slot_exit)

    def _create_widgets(self):
        parent = QWidget(self)
        self.setCentralWidget(parent)
        lv = QHBoxLayout(parent)

        self.tab_widget = QTabWidget(self)
        lv.addWidget(self.tab_widget)

        #image_widget = ImageWidget(self)
        #image_widget.set_image(torch.rand(3, 128, 128))
        #lh.addWidget(image_widget)
        #task_widget = TaskWidget(self, clipig=self.clipig)
        #lv.addWidget(task_widget)

        # lv.addStretch()

        self.status_label = QLabel(self)
        self.statusBar().addWidget(self.status_label)

    def close(self) -> bool:
        if not super().close():
            return False

        # self.slot_save_sessions()
        return True

    def slot_exit(self):
        self.close()

    def slot_new_task(self):
        task_widget = TaskWidget(self, clipig=self.clipig, preset_model=self.preset_model)
        task_widget.signal_run_task.connect(self.slot_run_task)
        task_widget.signal_stop_task.connect(self.slot_stop_task)
        self.tab_widget.addTab(task_widget, f"Task #{task_widget.task_id}")
        tab_index = self.tab_widget.count() - 1

        self._task_map[task_widget.task_id] = {
            "status": "undefined",
            "widget": task_widget,
            "tab_index": tab_index,
        }

    def slot_run_task(self, task_id: Hashable, config: dict):
        self.clipig.run_task(task_id, config)

    def slot_stop_task(self, task_id: Hashable):
        self.clipig.stop_task(task_id)

    def _slot_idle(self):
        for event in self.clipig.events(blocking=False):
            if event.get("task"):
                self._task_event(event["task"]["id"], event["task"])

        QTimer.singleShot(self._refresh_msec, self._slot_idle)

    def _task_event(self, task_id: Hashable, event: dict):
        # print("task_event", task_id, event.keys())
        task_data = self._task_map[task_id]

        if "status" in event:
            task_data["status"] = event["status"]
            self.tab_widget.setTabText(task_data["tab_index"], f"Task #{task_id} ({event['status']})")

        self.status_label.setText(
            f"Tasks: {len(self._task_map)}"
        )

        task_data["widget"].slot_task_event(event)
