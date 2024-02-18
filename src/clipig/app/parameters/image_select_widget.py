import json
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..dialogs import FileDialog


class ImageSelectWidget(QWidget):

    signal_value_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._do_not_emit_value_change = False
        self._create_widgets()

    def get_value(self) -> str:
        return self.input_widget.text()

    def set_value(self, value: str, emit: bool = False):
        try:
            self._do_not_emit_value_change = not emit
            self.input_widget.setText(value)
        finally:
            self._do_not_emit_value_change = False

    def _create_widgets(self):
        lh = QHBoxLayout(self)

        self.butt_select = QPushButton("...", self)
        self.butt_select.clicked.connect(self.show_dialog)
        lh.addWidget(self.butt_select)

        self.input_widget = QLineEdit(self)
        self.input_widget.textChanged.connect(self._value_changed)
        lh.addWidget(self.input_widget, stretch=10)

        model = QFileSystemModel(self)
        model.setFilter(QDir.Filter.AllEntries)
        #model.setNameFilters(["*.png", "*.jpg"])
        model.setRootPath(str(Path("~").expanduser()))
        self.completer = QCompleter(model, self)
        self.input_widget.setCompleter(self.completer)

    def _value_changed(self, v: str):

        if not self._do_not_emit_value_change:
            self.signal_value_changed.emit(self.get_value())

    def show_dialog(self):
        filename = FileDialog.get_load_filename(FileDialog.T_Image, parent=self)
        if filename:
            self.set_value(filename, emit=True)
