from functools import partial
from pathlib import Path
from typing import Dict, Hashable, Optional, List

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..clipig_worker import ClipigWorker
from .task.task_widget import TaskWidget
from .models.preset_model import PresetModel
from .images import LImage
from .project import ProjectWidget
from .dialogs import FileDialog


class MainWindow(QMainWindow):

    def __init__(self, clipig: ClipigWorker):
        super().__init__()

        self.clipig = clipig
        self._projects: List[ProjectWidget] = []
        self._refresh_msec = 300
        self.preset_model = PresetModel(self)

        self.setWindowTitle(self.tr("CLIPig ]["))
        # self.setWindowFlag(Qt.WindowMinMaxButtonsHint, True)

        self._create_main_menu()
        self._create_widgets()

        QTimer.singleShot(self._refresh_msec, self._slot_idle)

    def _create_main_menu(self):
        menu = self.menuBar().addMenu(self.tr("&File"))
        self._menu_project = self.menuBar().addMenu(self.tr("&Project"))

        menu.addAction(self.tr("&New Project"), self.slot_new_project, "CTRL+N")
        # menu.addAction(self.tr("New &Task"), self.slot_new_task)

        menu.addSeparator()

        menu.addAction(self.tr("&Open Project"), self.slot_load_project, "CTRL+O")

        self._action_save_project = action = QAction(self.tr("&Save Project"), self)
        action.setEnabled(False)
        action.setShortcut("CTRL+S")
        action.triggered.connect(self.slot_save_project)
        menu.addAction(action)
        self._action_save_project_as = action = QAction(self.tr("Save Project as ..."), self)
        action.setEnabled(False)
        action.setShortcut("CTRL+SHIFT+S")
        action.triggered.connect(self.slot_save_project_as)
        menu.addAction(action)

        menu.addSeparator()

        menu.addAction(self.tr("E&xit"), self.slot_exit)

    def _create_widgets(self):
        parent = QWidget(self)
        self.setCentralWidget(parent)
        lv = QVBoxLayout(parent)
        lv.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget(self)
        lv.addWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self._tab_changed)

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
        if self._projects:
            project_widget = self.slot_new_project()
        else:
            project_widget = self.current_project()

        return project_widget.slot_new_task()

    def slot_new_project(self):
        project_widget = ProjectWidget(self, clipig=self.clipig, preset_model=self.preset_model)

        self.tab_widget.addTab(project_widget, project_widget.project_name)
        tab_index = self.tab_widget.count() - 1

        project_widget.signal_changed.connect(partial(self._update_project_tab, project_widget))
        project_widget.signal_menu_changed.connect(self._update_project_menu)

        self._projects.append(project_widget)

        self.tab_widget.setCurrentIndex(tab_index)
        return project_widget

    def current_project(self) -> Optional[ProjectWidget]:
        index = self.tab_widget.currentIndex()
        if 0 <= index < len(self._projects):
            return self.tab_widget.widget(index)

    def get_project_tab_index(self, project_widget: ProjectWidget) -> Optional[int]:
        for i in range(self.tab_widget.count()):
            if self.tab_widget.widget(i) == project_widget:
                return i

    def select_project(self, project_widget: ProjectWidget):
        self.tab_widget.setCurrentWidget(project_widget)

    def _slot_idle(self):
        task_map: Optional[Dict[Hashable, dict]] = None

        for event in self.clipig.events(blocking=False):
            if event.get("task"):

                if 0:
                    if task_map is None:
                        task_map = {}
                        for project in self._projects:
                            task_map.update(project._task_map)

                    task_id = event["task"]["id"]
                    if task_id in task_map:
                        task_map[task_id]["widget"].slot_task_event(event["task"])
                else:
                    for proj in self._projects:
                        proj.task_event(event["task"]["id"], event["task"])

        QTimer.singleShot(self._refresh_msec, self._slot_idle)

    def slot_save_project(self):
        if project := self.current_project():
            if project.project_filename:
                project.save_project(project.project_filename)
            else:
                self.slot_save_project_as()

    def slot_save_project_as(self):
        if project := self.current_project():
            filename = FileDialog.get_save_filename(
                FileDialog.T_Project, parent=self,
            )
            if filename:
                project.save_project(filename)

    def slot_load_project(self):
        filename = FileDialog.get_load_filename(FileDialog.T_Project, parent=self)
        if filename:
            filename = Path(filename)

            for project_widget in self._projects:
                if project_widget.project_filename == filename:
                    self.select_project(project_widget)
                    return

            project = self.slot_new_project()
            project.load_project(filename)

    def _tab_changed(self):
        self._update_project_menu()

    def _update_project_menu(self):
        project = self.tab_widget.currentWidget()
        if not project:
            self._menu_project.clear()
            self._action_save_project.setEnabled(False)
            return

        self._action_save_project.setEnabled(True)
        self._menu_project.clear()
        project.add_menu_actions(self._menu_project)

    def _get_project_tab_text(self, project_widget: ProjectWidget) -> str:
        text = project_widget.project_name

        if project_widget.project_filename:
            text = f"{text} ({project_widget.project_filename.name})"

        if not project_widget.is_saved:
            text = f"{text} *"
        return text

    def _update_project_tab(self, project_widget: ProjectWidget):
        tab_index = self.get_project_tab_index(project_widget)
        if tab_index is not None:
            self.tab_widget.setTabText(tab_index, self._get_project_tab_text(project_widget))

