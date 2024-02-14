import json
import os
from pathlib import Path
from functools import partial
from typing import List
from pathlib import Path
from copy import deepcopy
from typing import Optional, Tuple

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage, LImageLayer


class LImageModel(QAbstractTableModel):

    COLUMNS = [
        "thumbnail", "selected", "active", "name", "transparency", "size", "position", "repeat",
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

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable

        if not index.isValid():
            return flags

        layer = self._limage.layers[index.row()]
        column = self.COLUMNS[index.column()]

        if column in ("thumbnail", ):
            flags |= Qt.ItemIsDragEnabled

        if column in ("name", "width", "height", "transparency", "repeat", "position"):
            flags |= Qt.ItemIsEditable

        return flags

    def data(self, index: QModelIndex, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return None

        layer = self._limage.layers[index.row()]
        column = self.COLUMNS[index.column()]

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            for_edit = role == Qt.ItemDataRole.EditRole

            if column == "name":
                return layer.name
            if column == "selected":
                return layer.selected
            if column == "active":
                return layer.active
            if column == "transparency":
                return layer.transparency
            if column == "repeat":
                return f"{layer.repeat[0]}x{layer.repeat[1]}"
            if column == "position":
                return f"{layer.position[0]}x{layer.position[1]}"

            if not for_edit:
                if column == "size":
                    s = layer.size()
                    return f"{s.width()}x{s.height()}"

        #elif role == Qt.ItemDataRole.SizeHintRole:
        #    return QSize(10, 10)

        elif role == Qt.ItemDataRole.DecorationRole:
            if column == "thumbnail":
                return layer.thumbnail()

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            flags = Qt.AlignmentFlag.AlignHCenter
            if column in ( "transparency", ):
                flags |= Qt.AlignmentFlag.AlignRight
            return flags

        elif role == Qt.ItemDataRole.UserRole:
            return layer

        elif role == Qt.ItemDataRole.BackgroundRole:
            pass

        elif role == Qt.ItemDataRole.FontRole:
            pass

    def setData(self, index: QModelIndex, value, role: Qt.ItemDataRole = ...):
        if not index.isValid():
            return

        layer: LImageLayer = self._limage.layers[index.row()]
        column = self.COLUMNS[index.column()]

        if column == "active":
            layer.set_active(value)
            return True

        elif column == "selected":
            layer.set_selected()
            return True

        elif column == "name":
            layer.set_name(value)
            return True

        elif column == "transparency":
            layer.set_transparency(value)
            return True

        elif column == "position":
            layer.set_position(parse_xy(value, default=(1, 1)))
            return True

        elif column == "repeat":
            layer.set_repeat(parse_xy(value, default=(1, 1)))
            return True

        return False

    def set_limage(self, limage: Optional[LImage] = None):
        self._limage = limage
        self.modelReset.emit()

    def set_table_delegates(self, table: QTableView):
        delegate = LImageModelCheckboxDelegate(self)
        table.setItemDelegateForColumn(self.COLUMNS.index("active"), delegate)
        table.setItemDelegateForColumn(self.COLUMNS.index("selected"), delegate)

        table.setColumnWidth(self.COLUMNS.index("selected"), 10)
        table.setColumnWidth(self.COLUMNS.index("active"), 10)
        table.setColumnWidth(self.COLUMNS.index("name"), 200)


class LImageModelCheckboxDelegate(QItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)

    def createEditor(self, parent: QWidget, option, index: QModelIndex):
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


def parse_xy(text: str, default: Tuple[int, int]) -> Tuple[int, int]:
    try:
        x, y = [int(t) for t in text.split("x")]
        return x, y
    except Exception:
        return default
