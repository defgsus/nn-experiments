import json
import os
from pathlib import Path
from functools import partial
from typing import List
from pathlib import Path
from copy import deepcopy
from typing import Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage, ImageLayer


class LImageModel(QAbstractTableModel):

    COLUMNS = [
        "thumbnail", "selected", "active", "name", "width", "height",
    ]

    def __init__(self, parent):
        super().__init__(parent)

        self._limage: Optional[LImage] = None

    def rowCount(self, parent = ...):
        return len(self._limage.layers) if self._limage else 0

    def columnCount(self, parent = ...):
        return len(self.COLUMNS)

    def parent(self, child):
        return QModelIndex()

    def index(self, row, column, parent = ...):
        if not (self._limage and 0 <= column < len(self.COLUMNS) and 0 <= row < len(self._limage.layers)):
            return QModelIndex()

        return self.createIndex(row, column)

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole = ...):
        if not (orientation == Qt.Horizontal and 0 <= section < len(self.COLUMNS)):
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section].replace("selected", "sel").replace("active", "act")

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return None

        layer = self._limage.layers[index.row()]
        column = self.COLUMNS[index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if column == "name":
                return layer.name
            elif column == "selected":
                return layer.selected
            elif column == "active":
                return layer.active
            elif column == "width":
                return layer.size().width()
                # return f"{size.width()}x{size.height()}"
            elif column == "height":
                return layer.size().height()

        elif role == Qt.ItemDataRole.EditRole:
            if column == "name":
                return layer.name
            elif column == "selected":
                return layer.active
            elif column == "active":
                return layer.active

        #elif role == Qt.ItemDataRole.SizeHintRole:
        #    return QSize(10, 10)

        elif role == Qt.ItemDataRole.DecorationRole:
            if column == "thumbnail":
                return layer.thumbnail()

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if column in ("width", "height"):
                return Qt.AlignRight

        elif role == Qt.ItemDataRole.UserRole:
            return layer

        elif role == Qt.ItemDataRole.BackgroundRole:
            pass

        elif role == Qt.ItemDataRole.FontRole:
            pass

    def setData(self, index: QModelIndex, value, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return

        layer = self._limage.layers[index.row()]
        column = self.COLUMNS[index.column()]

        if column == "active":
            layer.set_active(value)

        elif column == "selected":
            layer.set_selected()

    def set_limage(self, limage: Optional[LImage] = None):
        self._limage = limage
        self.modelReset.emit()

    def set_table_delegates(self, table: QTableView):
        table.setItemDelegate(LImageModelItemDelegate(self))
        table.setColumnWidth(self.COLUMNS.index("selected"), 10)
        table.setColumnWidth(self.COLUMNS.index("active"), 10)
        table.setColumnWidth(self.COLUMNS.index("name"), 200)


class LImageModelItemDelegate(QItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)

    def createEditor(self, parent: QWidget, option, index: QModelIndex):
        column = LImageModel.COLUMNS[index.column()]
        if column == "active":
            return QLineEdit(parent)

        return None

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        column = LImageModel.COLUMNS[index.column()]

        if column == "active":
            rect = option.rect.adjusted(5, 5, -5, -5)
            self.drawCheck(painter, option, rect, Qt.Checked if index.data() else Qt.Unchecked)

        elif column == "selected":
            if index.data():
                rect = option.rect.adjusted(5, 5, -5, -5)
                self.drawCheck(painter, option, rect, Qt.Checked)

        else:
            super().paint(painter, option, index)

    def editorEvent(self, event: QInputEvent, model: LImageModel, option, index: QModelIndex):
        column = LImageModel.COLUMNS[index.column()]

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:

            if column == "active":
                model.setData(index, not index.data())
                return True

            elif column in ("thumbnail", "selected"):
                col = LImageModel.COLUMNS.index("selected")
                model.setData(model.index(index.row(), col), True)

        return False
