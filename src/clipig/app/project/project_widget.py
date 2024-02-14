import tarfile
import os
from pathlib import Path
from typing import Dict, Hashable, Union, Optional

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from src.clipig.clipig_worker import ClipigWorker
from src.clipig.app.task.task_widget import TaskWidget
from src.clipig.app.models.preset_model import PresetModel
from src.clipig.app.images import LImageWidget, LImage
from src.clipig.app.dialogs import ParameterDialog

from src.util.files import Filestream


class ProjectWidget(QWidget):

    signal_changed = pyqtSignal()

    def __init__(
            self,
            *args,
            clipig: ClipigWorker,
            preset_model: PresetModel,
            **kwargs,
    ):
        super().__init__(*args, *kwargs)

        self.project_name = "new project"
        self.project_filename: Optional[Path] = None

        self.clipig = clipig
        self._task_map: Dict[Hashable, dict] = {}
        self._is_saved = False
        self._refresh_msec = 300
        self.preset_model = preset_model

        self._create_widgets()

    @property
    def is_saved(self):
        return self._is_saved

    def set_changed(self):
        self._is_saved = False
        self.signal_changed.emit()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget(self)
        lv.addWidget(self.tab_widget)

    def add_menu_actions(self, menu: QMenu):
        menu.addAction(self.tr("New &Task"), self.slot_new_task, "CTRL+T")
        menu.addSeparator()
        menu.addAction(self.tr("Rename ..."), self.slot_rename)

    def get_settings(self) -> dict:
        return {
            "name": self.project_name,
        }

    def set_settings(self, settings: dict):
        self.project_name = settings["name"]
        self.signal_changed.emit()

    def slot_new_task(self) -> TaskWidget:
        task_widget = TaskWidget(self, clipig=self.clipig, preset_model=self.preset_model)
        task_widget.signal_changed.connect(self.set_changed)
        task_widget.signal_run_task.connect(self.slot_run_task)
        task_widget.signal_stop_task.connect(self.slot_stop_task)
        task_widget.signal_new_task_with_image.connect(self.slot_new_task_with_image)
        self.tab_widget.addTab(task_widget, f"Task #{task_widget.task_id}")
        tab_index = self.tab_widget.count() - 1

        self._task_map[task_widget.task_id] = {
            "status": "undefined",
            "widget": task_widget,
            # TODO: this is not a sustainable idea
            "tab_index": tab_index,
        }

        self.tab_widget.setCurrentIndex(tab_index)
        self.set_changed()
        return task_widget

    def slot_new_task_with_image(self, limage: LImage):
        new_widget = self.slot_new_task()
        new_widget.set_limage(limage)

    def slot_run_task(self, task_id: Hashable, config: dict):
        self.clipig.run_task(task_id, config)

    def slot_stop_task(self, task_id: Hashable):
        self.clipig.stop_task(task_id)

    def task_event(self, task_id: Hashable, event: dict):
        # print("task_event", task_id, event.keys())
        task_data = self._task_map[task_id]

        if "status" in event:
            task_data["status"] = event["status"]
            self.tab_widget.setTabText(task_data["tab_index"], f"Task #{task_id} ({event['status']})")

        self.status_label.setText(
            f"Tasks: {len(self._task_map)}"
        )

        task_data["widget"].slot_task_event(event)

    def save_project(self, filename: Union[str, Path]):
        filename = Path(filename)

        config_data = {
            "project_settings": self.get_settings(),
            "task_config_filenames": [
                f"task_{i:02}/config.yaml"
                for i, task in enumerate(self._task_map.values())
            ],
        }

        os.makedirs(filename.parent, exist_ok=True)
        with Filestream(filename, "w") as filestream:

            filestream.write_yaml("config.yaml", config_data)

            for i, task in enumerate(self._task_map.values()):
                task["widget"].save_to_filestream(filestream, f"task_{i:02}")

        self._is_saved = True
        self.project_filename = filename
        self.signal_changed.emit()

    def load_project(self, filename: Union[str, Path]):
        filename = Path(filename)

        with Filestream(filename, "r") as filestream:

            config_data = filestream.read_yaml("config.yaml")

            for task_filename in config_data["task_config_filenames"]:

                task_widget = self.slot_new_task()
                task_widget.load_from_filestream(filestream, task_filename)

            self.set_settings(config_data["project_settings"])

        self._is_saved = True
        self.project_filename = filename
        self.signal_changed.emit()

    def slot_rename(self):
        accepted, name = ParameterDialog.get_string_value(
            self.project_name, name="project name",
        )
        if accepted:
            self.project_name = name
            self.set_changed()
