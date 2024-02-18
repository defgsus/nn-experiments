from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage, LImageLayer
from .limage_model import LImageModel
from ..dialogs import ResizeDialog


class LImageLayersWidget(QWidget):

    signal_new_layer = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._limage: Optional[LImage] = None

        self._create_widgets()
        self.set_limage(None)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self.table_widget = QTableView()
        lv.addWidget(self.table_widget)
        self.table_widget.setShowGrid(False)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setMaximumHeight(160)
        self.table_widget.doubleClicked.connect(self._table_dbl_click)

    def set_limage(self, limage: Optional[LImage] = None):
        self._limage = limage

        if self._limage is None:
            self.table_widget.setModel(None)
        else:
            model: LImageModel = self._limage.get_model()
            self.table_widget.setModel(model)
            model.set_table_delegates(self.table_widget)
            #self.table_widget.resizeColumnsToContents()

    def _context_menu(self, pos: QPoint):
        if self._limage is None:
            return

        menu = QMenu()
        self._limage.add_menu_actions(menu)
        #menu.addAction(self.tr("Add layer"), self.signal_new_layer)

        menu.addSeparator()

        indices = self.table_widget.selectedIndexes()
        rows = set(i.row() for i in indices)
        if indices:
            menu.addAction(
                self.tr("Remove Layers") if len(rows) > 1 else self.tr("Remove Layer"),
                partial(self._delete_rows, list(rows))
            )

        menu.exec(self.mapToGlobal(pos))

    def _delete_rows(self, rows: List[int]):
        self._limage.delete_layers(rows)

    def _table_dbl_click(self, index: QModelIndex):
        size_column = LImageModel.COLUMNS.index("size")
        if index.column() == size_column:
            layer = index.data(Qt.ItemDataRole.UserRole)
            result = ResizeDialog.run_dialog(
                size=layer.size(),
                title=f"Resize {layer.name}"
            )
            if result is not None:
                layer.set_image_size(**result._asdict())
