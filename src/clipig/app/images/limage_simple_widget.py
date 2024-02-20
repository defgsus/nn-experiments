from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..util import image_to_qimage, AnyImage
from .limage import LImage, LImageLayer
from .limage_canvas_widget import LImageCanvasWidget
from .limage_layers_widget import LImageLayersWidget
from ..dialogs import FileDialog
from .tools_widget import ImageToolsWidget
from .color_palette import ColorPaletteWidget
from . import image_tools


class LImageSimpleWidget(QWidget):

    signal_changed = pyqtSignal()

    def __init__(self, *, project: "ProjectWidget", parent: Optional[QWidget] = None):
        super().__init__(parent or project)

        from ..project import ProjectWidget
        self._project: ProjectWidget = project

        self._limage = LImage()

        self._create_widgets()

        self._limage.get_model().dataChanged.connect(lambda *args, **kwargs: self._set_changed())
        self._limage.get_model().modelReset.connect(lambda *args, **kwargs: self._set_changed())

        self.setMinimumSize(200, 200)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    @property
    def limage(self):
        return self._limage

    def get_settings(self) -> dict:
        return {
            "canvas_settings": self.canvas.get_settings(),
        }

    def set_settings(self, settings: dict):
        self.canvas.set_settings(settings["canvas_settings"])

    def set_limage(self, image: LImage):
        self._limage = image

        self.canvas.set_limage(self._limage)
        self._limage.get_model().dataChanged.connect(lambda *args, **kwargs: self._set_changed())
        self._limage.get_model().modelReset.connect(lambda *args, **kwargs: self._set_changed())
        self._set_changed()

    def set_image(self, name: str, image: AnyImage):
        """
        Create or replace the layer matching `name` with the given image
        """
        image = image_to_qimage(image)

        if layer := self._limage.get_layer(name):
            layer.set_image(image)
        else:
            self._limage.add_layer(image=image, name=name)

        self._set_changed()

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea(self)
        lv.addWidget(self.scroll_area, stretch=100)

        self.canvas = LImageCanvasWidget(self)
        self.canvas.set_limage(self._limage)
        self.scroll_area.setWidget(self.canvas)

        lv.addWidget(self.canvas.create_controls())

    def _context_menu(self, pos: QPoint):
        menu = QMenu()
        if self._limage is not None:
            if not self._limage.size().isEmpty():
                menu.addAction(self.tr("Save image as ..."), self.action_save_image_as)
            self._limage.add_menu_actions(menu, project=self._project)

        menu.exec(self.mapToGlobal(pos))

    def action_save_image_as(self):
        if self._limage is not None:

            # copy the current image before opening dialog
            image = self._limage.to_qimage()

            filename = FileDialog.get_save_filename(FileDialog.T_Image, parent=self)
            if filename:
                image.save(str(filename))

    def _set_changed(self):
        self.signal_changed.emit()
