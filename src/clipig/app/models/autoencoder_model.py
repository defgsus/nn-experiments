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

import yaml


class AutoencoderModel(QAbstractItemModel):

    autoencoder_directory = Path(__file__).resolve().parent.parent.parent / "models/autoencoder"

    def __init__(self, parent):
        super().__init__(parent)

        self._models = []
        self._scan_files()

    def rowCount(self, parent = ...):
        return len(self._models)

    def columnCount(self, parent = ...):
        return 1

    def parent(self, child):
        return QModelIndex()

    def index(self, row, column, parent = ...):
        if row < 0 or row >= len(self._models):
            return QModelIndex()
        if column != 0:
            return QModelIndex()

        return self.createIndex(row, column)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return None

        data = self._models[index.row()]

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
        return [m["name"] for m in self._models]

    def _scan_files(self):
        for filename in self.autoencoder_directory.rglob("**/*.y?ml"):
            try:
                with filename.open() as fp:
                    data = yaml.safe_load(fp)
            except Exception:
                continue

            self._models.append({
                "filename": filename,
                "name": str(filename.relative_to(self.autoencoder_directory)).rsplit(".", 1)[0],
                "config": data,
            })

        self._sort()

    def _sort(self):
        self._models.sort(key=lambda p: p["name"])
