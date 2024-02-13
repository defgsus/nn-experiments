from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .limage import LImage, ImageLayer
from .limage_model import LImageModel


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

    def set_limage(self, limage: Optional[LImage] = None):
        self._limage = limage

        if self._limage is None:
            self.table_widget.setModel(None)
        else:
            model = self._limage.get_model()
            self.table_widget.setModel(model)
            model.set_table_delegates(self.table_widget)
            #self.table_widget.resizeColumnsToContents()


    def _context_menu(self, pos: QPoint):
        if self._limage is None:
            return

        menu = QMenu()

        menu.addAction(self.tr("Add layer"), self.signal_new_layer)

        menu.exec(self.mapToGlobal(pos))
