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


class PresetModel(QAbstractItemModel):

    preset_directory: Path = Path(__file__).resolve().parent.parent.parent / "presets"

    def __init__(self, parent):
        super().__init__(parent)

        self._presets = []
        self._scan_files()

    def rowCount(self, parent = ...):
        return len(self._presets)

    def columnCount(self, parent = ...):
        return 1

    def parent(self, child):
        return QModelIndex()

    def index(self, row, column, parent = ...):
        if row < 0 or row >= len(self._presets):
            return QModelIndex()
        if column != 0:
            return QModelIndex()

        return self.createIndex(row, column)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return None

        preset = self._presets[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return preset["name"]

        elif role == Qt.ItemDataRole.UserRole:
            return preset

        elif role == Qt.ItemDataRole.BackgroundRole:
            pass

        elif role == Qt.ItemDataRole.FontRole:
            pass

    def default_config(self) -> dict:
        return self.data(self.index(0, 0), Qt.UserRole)

    def preset_names(self) -> List[str]:
        return [p["name"] for p in self._presets]

    def _scan_files(self):

        for filename in self.preset_directory.rglob("**/*.y?ml"):
            try:
                with filename.open() as fp:
                    data = yaml.safe_load(fp)
            except Exception:
                continue

            self._presets.append({
                "filename": filename,
                "name": str(filename.relative_to(self.preset_directory)).rsplit(".", 1)[0],
                "config": data,
            })

        self._sort()

    def _sort(self):
        self._presets.sort(key=lambda p: p["name"])
        self._presets.sort(key=lambda p: 0 if p["name"] == "default" else 1)

    def save_preset(self, name: str, config: dict):
        filename = self.preset_directory / f"{name}.yaml"

        os.makedirs(filename.parent, exist_ok=True)
        with open(filename, "wt") as fp:
            yaml.safe_dump(config, fp)

        for p in self._presets:
            if p["name"] == name:
                p["config"] = config
                self.modelReset.emit()
                return

        self._presets.append({
            "filename": filename,
            "name": name,
            "config": config,
        })

        self.modelReset.emit()
