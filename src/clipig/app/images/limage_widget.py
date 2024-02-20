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


class LImageWidget(QWidget):

    signal_new_task_with_image = pyqtSignal(LImage)
    signal_changed= pyqtSignal()

    def __init__(self, *, project: "ProjectWidget", parent: Optional[QWidget] = None):
        super().__init__(parent or project)

        from ..project import ProjectWidget
        self._project: ProjectWidget = project

        self._limage = LImage()
        self._tool: Optional[image_tools.ImageToolBase] = None

        self._last_save_directory: Optional[str] = None
        self._last_loaded_images: List[str] = []

        self._create_widgets()

        self._limage.get_model().dataChanged.connect(lambda *args, **kwargs: self._set_changed())
        self._limage.get_model().modelReset.connect(lambda *args, **kwargs: self._set_changed())

        self.setMinimumSize(200, 200)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    @property
    def limage(self):
        return self._limage

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)

        self.tools_widget = ImageToolsWidget(self)
        self.tools_widget.signal_tool_changed.connect(self.set_tool)
        lv.addWidget(self.tools_widget)

        lh = QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(lh, stretch=100)

        self.color_palette_widget = ColorPaletteWidget(self)
        lh.addWidget(self.color_palette_widget)
        self.color_palette_widget.signal_color_changed.connect(self.tools_widget.set_color)

        self.scroll_area = QScrollArea(self)
        lh.addWidget(self.scroll_area, stretch=100)

        self.canvas = LImageCanvasWidget(self)
        self.canvas.set_limage(self._limage)
        self.scroll_area.setWidget(self.canvas)
        self.canvas.signal_mouse_event.connect(self._canvas_mouse_event)

        lv.addWidget(self.canvas.create_controls())

        self.layers_widget = LImageLayersWidget(parent=self, project=self._project)
        self.layers_widget.set_limage(self._limage)
        lv.addWidget(self.layers_widget)
        self.layers_widget.signal_new_layer.connect(self.action_new_layer)

        self.info_label = QLabel(self)
        lv.addWidget(self.info_label)

    def get_settings(self) -> dict:
        return {
            "canvas_settings": self.canvas.get_settings(),
        }

    def set_settings(self, settings: dict):
        self.canvas.set_settings(settings["canvas_settings"])

    def set_limage(self, image: LImage):
        self._limage = image

        self.canvas.set_limage(self._limage)
        self.layers_widget.set_limage(self._limage)
        if self._tool is not None:
            self._tool = self._tool.__class__(limage=self._limage, config=self._tool.config)
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

    def _context_menu(self, pos: QPoint):
        menu = QMenu()
        if self._limage is not None:
            if not self._limage.size().isEmpty():
                menu.addAction(self.tr("Save image as ..."), self.action_save_image_as)
            menu.addAction(self.tr("To new Task"), partial(self.action_to_new_task, False))
            menu.addAction(self.tr("To new Task (merged)"), partial(self.action_to_new_task, True))
            self._limage.add_menu_actions(menu, project=self._project)

        menu.addSeparator()
        menu.addAction(self.tr("Open image ..."), self.action_load_image_dialog)
        for filename in self._last_loaded_images:
            menu.addAction(
                self.tr("Reload {filename}").format(filename=Path(filename).name),
                partial(self.action_load_image, filename),
            )

        menu.exec(self.mapToGlobal(pos))

    def action_save_image_as(self):
        if self._limage is not None:

            # copy the current image before opening dialog
            image = self._limage.to_qimage()

            filename = FileDialog.get_save_filename(FileDialog.T_Image, parent=self)
            if filename:
                image.save(str(filename))

    def action_load_image_dialog(self):
        filename = FileDialog.get_load_filename(FileDialog.T_Image, parent=self)
        if filename:
            self.action_load_image(filename)

            if filename in self._last_loaded_images:
                self._last_loaded_images.remove(filename)
            self._last_loaded_images.insert(0, filename)
            self._last_loaded_images = self._last_loaded_images[:4]

    def action_load_image(self, filename: Union[str, Path]):
        image = QImage(filename)
        name = Path(filename).name
        self.set_image(name, image)

    def action_new_layer(self):
        self._limage.add_layer()
        self._set_changed()

    def action_to_new_task(self, merged: bool = False):
        self.signal_new_task_with_image.emit(self._limage.copy(merged=merged))

    def _set_changed(self):
        rect = self._limage.rect()
        self.info_label.setText(
            f"{rect.width()}x{rect.height()}"
        )

        self.signal_changed.emit()

    def set_tool(self, tool_name: str, config: Optional[dict] = None):
        klass = image_tools.image_tools.get(tool_name)
        if not klass:
            return

        self._tool = klass(limage=self._limage, config=config or {})

    def _canvas_mouse_event(self, event: image_tools.MouseEvent):
        if self._tool is not None:
            self._tool.mouse_event(event)
