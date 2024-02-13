from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..util import image_to_qimage, AnyImage
from .limage import LImage, ImageLayer
from .limage_canvas_widget import LImageCanvasWidget
from .limage_layers_widget import LImageLayersWidget


class LImageWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.limage = LImage()
        self.limage.add_layer("bla", QImage("/home/bergi/Pictures/__diverse/swall03.jpg"))

        self._last_save_directory: Optional[str] = None
        self._last_loaded_images: List[str] = []

        self._create_widgets()

        self.limage.get_model().dataChanged.connect(lambda *args, **kwargs: self._update_limage())
        self.limage.get_model().modelReset.connect(lambda *args, **kwargs: self._update_limage())

        self.setMinimumSize(200, 200)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        self.setLayout(lv)

        self.scroll_area = QScrollArea(self)
        lv.addWidget(self.scroll_area, stretch=100)

        self.canvas = LImageCanvasWidget(self)
        self.canvas.set_limage(self.limage)
        self.scroll_area.setWidget(self.canvas)

        lv.addWidget(self.canvas.create_controls())

        self.layers_widget = LImageLayersWidget(self)
        self.layers_widget.set_limage(self.limage)
        lv.addWidget(self.layers_widget)
        self.layers_widget.signal_new_layer.connect(self.action_new_layer)

        self.info_label = QLabel(self)
        lv.addWidget(self.info_label)

    def set_image(self, name: str, image: AnyImage):
        image = image_to_qimage(image)

        if layer := self.limage.get_layer(name):
            layer.set_image(image)
        else:
            self.limage.add_layer(image=image, name=name)

        self._update_limage()

    def _context_menu(self, pos: QPoint):
        menu = QMenu()
        if self.limage is not None:
            menu.addAction(self.tr("Save image as ..."), self.action_save_image_as)

        menu.addSeparator()
        menu.addAction(self.tr("Load image ..."), self.action_load_image_dialog)
        for filename in self._last_loaded_images:
            menu.addAction(
                self.tr("Reload {filename}").format(Path(filename).name),
                partial(self.action_load_image, filename),
            )

        menu.exec(self.mapToGlobal(pos))

    def action_save_image_as(self):
        if self.limage is not None:

            # copy the current image before opening dialog
            image = self.limage.to_qimage()

            filename, _ = QFileDialog.getSaveFileName(
                parent=self,
                caption=self.tr("Save image"),
                filter="*.png",
                directory=self._last_save_directory,
            )
            if filename:
                filename = Path(filename)
                self._last_save_directory = str(filename.parent)

                if not filename.suffix:
                    filename = filename.with_suffix(".png")

                image.save(str(filename))

    def action_load_image_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption=self.tr("Load image"),
            filter="*.png",
        )
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
        self.limage.add_layer()
        self._update_limage()

    def _update_limage(self):
        rect = self.limage.rect()
        self.info_label.setText(
            f"{rect.width()}x{rect.height()}"
        )
