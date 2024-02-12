import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from src.clipig.task_parameters import get_task_parameters


class NewTransformationDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(self.tr("new transformation"))
        self.default_parameters = get_task_parameters()

        lv = QVBoxLayout(self)

        self.list_widget = QListWidget(self)
        lv.addWidget(self.list_widget)

        for trans_name in self.default_parameters["transformations"]:
            self.list_widget.addItem(trans_name)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    @classmethod
    def get_transformation_name(cls, parent: QWidget) -> Optional[str]:
        dialog = cls(parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            if dialog.list_widget.currentRow() >= 0:
                return dialog.list_widget.currentItem().text()
