from typing import Dict, Hashable, Optional

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..clipig_worker import ClipigWorker
from .task.task_widget import TaskWidget
from .models.preset_model import PresetModel
from .images import LImage
from .project import ProjectWidget


class MainWindow(QMainWindow):

    def __init__(self, clipig: ClipigWorker):
        super().__init__()

        self.clipig = clipig
        self._project_map: Dict[Hashable, dict] = {}
        self._refresh_msec = 300
        self.preset_model = PresetModel(self)

        self.setWindowTitle(self.tr("CLIPig ]["))
        # self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)

        self._create_main_menu()
        self._create_widgets()
        QTimer.singleShot(self._refresh_msec, self._slot_idle)

    def _create_main_menu(self):
        menu = self.menuBar().addMenu(self.tr("&File"))

        project = self.current_project()

        menu.addAction(self.tr("New &Project"), self.slot_new_project)
        menu.addAction(self.tr("New &Task"), self.slot_new_task)

        menu.addSeparator()

        if project:
            menu.addAction(self.tr("Save Project"), self.slot_save_project, "CTRL+S")

            menu.addSeparator()

        menu.addAction(self.tr("E&xit"), self.slot_exit)

    def _create_widgets(self):
        parent = QWidget(self)
        self.setCentralWidget(parent)
        lv = QVBoxLayout(parent)
        lv.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget(self)
        lv.addWidget(self.tab_widget)

        self.status_label = QLabel(self)
        self.statusBar().addWidget(self.status_label)

    def close(self) -> bool:
        if not super().close():
            return False

        # self.slot_save_sessions()
        return True

    def slot_exit(self):
        self.close()

    def slot_new_task(self) -> TaskWidget:
        if not self._project_map:
            project_widget = self.slot_new_project()
        else:
            project_widget = self.current_project()

        return project_widget.slot_new_task()

    def slot_new_project(self):
        project_name = "proj"
        project_widget = ProjectWidget(self, clipig=self.clipig, preset_model=self.preset_model)

        self.tab_widget.addTab(project_widget, project_name)
        tab_index = self.tab_widget.count() - 1

        self._project_map[project_widget] = {
            "widget": project_widget,
            "tab_index": tab_index,
        }

        self.tab_widget.setCurrentIndex(tab_index)
        return project_widget

    def current_project(self) -> Optional[ProjectWidget]:
        index = self.tab_widget.currentIndex()
        if 0 <= index < len(self._project_map):
            return self.tab_widget.widget(index)

    def _slot_idle(self):
        task_map: Optional[Dict[Hashable, dict]] = None

        for event in self.clipig.events(blocking=False):
            if event.get("task"):

                if task_map is None:
                    task_map = {}
                    for project in self._project_map.values():
                        task_map.update(project["widget"]._task_map)

                task_id = event["task"]["id"]
                if task_id in task_map:
                    self._task_event(task_id, event["task"])

        QTimer.singleShot(self._refresh_msec, self._slot_idle)
