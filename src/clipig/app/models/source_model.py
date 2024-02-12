import json
import os
from pathlib import Path
from functools import partial
from typing import List
from pathlib import Path
from copy import deepcopy
import urllib.parse

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from src.clipig import source_models


class SourceModelModel(QAbstractItemModel):

    def __init__(self, parent):
        super().__init__(parent)

        self._source_models = [
            {"name": name, "params": klass.PARAMS}
            for name, klass in source_models.source_models.items()
        ]

    def rowCount(self, parent = ...):
        return len(self._source_models)

    def columnCount(self, parent = ...):
        return 1

    def parent(self, child):
        return QModelIndex()

    def index(self, row, column, parent = ...):
        if row < 0 or row >= len(self._source_models):
            return QModelIndex()
        if column != 0:
            return QModelIndex()

        return self.createIndex(row, column)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return None

        data = self._source_models[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return data["name"]

        elif role == Qt.ItemDataRole.UserRole:
            return deepcopy(data)

        elif role == Qt.ItemDataRole.BackgroundRole:
            pass

        elif role == Qt.ItemDataRole.FontRole:
            pass

    def default_model(self) -> dict:
        return self.data(self.index(0, 0), Qt.UserRole)

    def model_names(self) -> List[str]:
        return [m["name"] for m in self._source_models]
