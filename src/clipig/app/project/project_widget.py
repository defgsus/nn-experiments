import io
import tarfile
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Hashable, Union, Optional, Callable, List

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from src.clipig.clipig_worker import ClipigWorker
from src.clipig.app.task.task_widget import TaskWidget
from src.clipig.app.models.preset_model import PresetModel
from src.clipig.app.images import LImage
from src.clipig.app.dialogs import ParameterDialog

from src.util.files import Filestream


class UndoAction:

    def __init__(
            self,
            name: str,
            undo: Callable,
            redo: Callable,
    ):
        self.name = name
        self.undo = undo
        self.redo = redo

    def __str__(self):
        return f"{self.__class__.__name__}(\"{self.name}\")"


class ProjectWidget(QWidget):

    signal_changed = pyqtSignal()
    signal_menu_changed = pyqtSignal()

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

        self._dialog_settings = {}
        self._undo_actions: List[UndoAction] = []
        self._undo_index = 0
        self._undo_action: Optional[QAction] = None
        self._redo_action: Optional[QAction] = None
        
        self._create_widgets()

    @property
    def is_saved(self):
        return self._is_saved

    def set_changed(self):
        self._is_saved = False
        self.signal_changed.emit()

    def set_menu_changed(self):
        self.signal_menu_changed.emit()

    def get_settings(self) -> dict:
        return {
            "name": self.project_name,
            "dialog_settings": deepcopy(self._dialog_settings),
        }

    def set_settings(self, settings: dict):
        self.project_name = settings["name"]
        if sett := settings.get("dialog_settings"):
            self._dialog_settings = deepcopy(sett)
        self.signal_changed.emit()

    def get_dialog_settings(self, name: str) -> dict:
        return deepcopy(self._dialog_settings.get(name) or {})

    def set_dialog_settings(self, name: str, settings: dict):
        if settings != self._dialog_settings.get("name"):
            self._dialog_settings[name] = deepcopy(settings)
            self.signal_changed.emit()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget(self)
        lv.addWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.set_menu_changed)

    def push_undo_action(self, name: str, undo: Callable, redo: Callable):
        self._undo_actions = self._undo_actions[:self._undo_index + 1]
        self._undo_index = len(self._undo_actions)
        self._undo_actions.append(UndoAction(name, undo, redo))
        self._update_undo_qactions()
    
    def _update_undo_qactions(self):
        if self._undo_action is not None:
            if self._undo_index < 0 or self._undo_index >= len(self._undo_actions):
                self._undo_action.setText("Undo")
                self._undo_action.setEnabled(False)
            else:
                self._undo_action.setText(f"Undo {self._undo_actions[self._undo_index].name}")
                self._undo_action.setEnabled(True)
        if self._redo_action is not None:
            if self._undo_index >= len(self._undo_actions) - 1:
                self._redo_action.setText("Redo")
                self._redo_action.setEnabled(False)
            else:
                self._redo_action.setText(f"Redo {self._undo_actions[self._undo_index + 1].name}")
                self._redo_action.setEnabled(True)

    def add_menu_actions(self, menu: QMenu):
        self._undo_action = action = QAction("Undo", self)
        action.triggered.connect(self.slot_undo)
        action.setShortcut("CTRL+Z")
        menu.addAction(action)
        self._redo_action = action = QAction("Redo", self)
        action.triggered.connect(self.slot_redo)
        action.setShortcut("CTRL+SHIFT+Z")
        menu.addAction(action)
        self._update_undo_qactions()

        menu.addSeparator()
        menu.addAction(self.tr("New &Task"), self.slot_new_task, "CTRL+T")
        if self.tab_widget.count():
            menu.addAction(self.tr("&Clone Task"), self.slot_copy_task)
            menu.addAction(self.tr("Delete Task"), self.slot_delete_task)
        menu.addSeparator()
        menu.addAction(self.tr("Rename Project ..."), self.slot_rename)
    
    def slot_undo(self):
        if 0 <= self._undo_index < len(self._undo_actions):
            self._undo_actions[self._undo_index].undo()
            self._undo_index -= 1
            self._update_undo_qactions()

    def slot_redo(self):
        if self._undo_index < len(self._undo_actions) - 1:
            self._undo_actions[self._undo_index + 1].redo()
            self._undo_index += 1
            self._update_undo_qactions()

    def slot_new_task(self) -> TaskWidget:
        task_widget = TaskWidget(self, project=self)
        task_widget.signal_changed.connect(self.set_changed)
        task_widget.signal_run_task.connect(self.slot_run_task)
        task_widget.signal_stop_task.connect(self.slot_stop_task)
        task_widget.signal_new_task_with_image.connect(lambda x: self.slot_new_task_with(limage=x))
        task_widget.signal_copy_task.connect(self.slot_copy_task)

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

    def slot_new_task_with(
            self,
            limage: Optional[LImage] = None,
            task_config: Optional[dict] = None,
    ):
        new_widget = self.slot_new_task()
        if limage is not None:
            new_widget.set_limage(limage)
        if task_config is not None:
            new_widget.set_task_config(task_config)
        self.set_changed()

    def slot_run_task(self, task_id: Hashable, config: dict):
        self.clipig.run_task(task_id, config)

    def slot_stop_task(self, task_id: Hashable):
        self.clipig.stop_task(task_id)

    def slot_copy_task(self):
        task: TaskWidget = self.tab_widget.currentWidget()
        buffer = io.BytesIO()
        with Filestream(buffer, "w") as fs:
            task.save_to_filestream(fs, ".")

        new_widget = self.slot_new_task()
        buffer.seek(0)
        with Filestream(buffer, "r") as fs:
            new_widget.load_from_filestream(fs, "config.yaml")

    def slot_delete_task(self):
        task_idx = self.tab_widget.currentIndex()
        task_id = self.tab_widget.widget(task_idx).task_id
        self.tab_widget.removeTab(task_idx)
        if task_id in self._task_map:
            del self._task_map[task_id]
        self.set_changed()

    def task_event(self, task_id: Hashable, event: dict):
        if task_id not in self._task_map:
            return

        # print("task_event", task_id, event.keys())
        task_data = self._task_map[task_id]

        if "status" in event:
            task_data["status"] = event["status"]
            self.tab_widget.setTabText(task_data["tab_index"], f"Task #{task_id} ({event['status']})")

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
