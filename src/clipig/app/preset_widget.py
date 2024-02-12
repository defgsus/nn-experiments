import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .preset_model import PresetModel


class PresetWidget(QWidget):

    signal_preset_changed = pyqtSignal(dict)
    signal_save_preset = pyqtSignal(str)

    def __init__(self, *args, preset_model: PresetModel, **kwargs):
        super().__init__(*args, **kwargs)

        self.preset_model = preset_model

        self._create_widgets()

    def _create_widgets(self):
        lh = QHBoxLayout(self)

        self.preset_list = QComboBox(self)
        self.preset_list.setModel(self.preset_model)
        lh.addWidget(self.preset_list)
        self.preset_list.currentIndexChanged.connect(self._index_changed)
        
        self.butt_load = QPushButton(self.tr("load"), self)
        self.butt_load.clicked.connect(self._load)
        lh.addWidget(self.butt_load)

        self.butt_save = QPushButton(self.tr("save"), self)
        self.butt_save.clicked.connect(self._save)
        lh.addWidget(self.butt_save)

        self.butt_save_as = QPushButton(self.tr("save as ..."), self)
        self.butt_save_as.clicked.connect(self._save_as)
        lh.addWidget(self.butt_save_as)

    def _index_changed(self, index: int):
        index = self.preset_model.index(index, 0)
        self.butt_load.setEnabled(index.isValid())
        self.butt_save.setEnabled(index.isValid())
        self.butt_save_as.setEnabled(index.isValid())

    def _load(self):
        index = self.preset_model.index(self.preset_list.currentIndex(), 0)
        if index.isValid():
            preset = self.preset_model.data(index, role=Qt.UserRole)
            self.signal_preset_changed.emit(preset["config"])

    def _save(self):
        index = self.preset_model.index(self.preset_list.currentIndex(), 0)
        if index.isValid():
            preset = self.preset_model.data(index, role=Qt.UserRole)
            self.signal_save_preset.emit(preset["name"])

    def _save_as(self):
        name = PresetNameDialog.get_preset_name(self, self.preset_model)
        if name:
            self.signal_save_preset.emit(name)


class PresetNameDialog(QDialog):
    def __init__(self, *args, preset_model: PresetModel, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(self.tr("new preset name"))
        self.preset_model = preset_model
        lv = QVBoxLayout(self)

        self.completer = QCompleter(self.preset_model.preset_names(), self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)

        self.input = QLineEdit(self)
        self.input.setCompleter(self.completer)
        self.input.setMinimumWidth(200)
        lv.addWidget(self.input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    @classmethod
    def get_preset_name(cls, parent: QWidget, preset_model: PresetModel) -> Optional[str]:
        dialog = cls(parent, preset_model=preset_model)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.input.text()
