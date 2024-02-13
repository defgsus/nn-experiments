from functools import partial
from pathlib import Path
from typing import Optional, List, Union

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .util import image_to_qimage


class ImageWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ignore_zoom_bar = False
        self.image: Optional[QImage] = None

        self._last_save_directory: Optional[str] = None
        self._last_loaded_images: List[str] = []

        self.setMinimumSize(200, 200)

        self._create_widgets()

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        self.setLayout(lv)

        self.scroll_area = QScrollArea(self)
        lv.addWidget(self.scroll_area)

        self.canvas = ImageDisplayCanvas(self)
        self.scroll_area.setWidget(self.canvas)

        self.info_label = QLabel(self)
        lv.addWidget(self.info_label)

        lh = QHBoxLayout()
        lv.addLayout(lh)

        self.repeat_input = QSpinBox(self)
        lh.addWidget(self.repeat_input)
        self.repeat_input.setRange(1, 10)
        self.repeat_input.setValue(self.repeat)
        self.repeat_input.setToolTip(self.tr("repeat"))

        self.repeat_input.setStatusTip(self.tr("repeat"))
        self.repeat_input.valueChanged.connect(self.set_repeat)

        self.zoom_bar = QScrollBar(Qt.Horizontal, self)
        lh.addWidget(self.zoom_bar)
        self.zoom_bar.setStatusTip(self.tr("zoom"))
        self.zoom_bar.setRange(1, 500)
        self.zoom_bar.setValue(self.zoom)
        self.zoom_bar.valueChanged.connect(self._zoom_bar_changed)
        self.zoom_bar.setToolTip(self.tr("zoom"))

        for zoom in (50, 100, 200, 300, 500):
            b = QPushButton(self.tr(f"{zoom}%"))
            if zoom == 100:
                font = b.font()
                font.setBold(True)
                b.setFont(font)
            b.setToolTip(self.tr("set zoom to {zoom}%").format(zoom=zoom))
            lh.addWidget(b)
            b.clicked.connect(partial(self.set_zoom, zoom))

    @property
    def zoom(self):
        return self.canvas.zoom

    @property
    def repeat(self):
        return self.canvas.num_repeat

    def set_zoom(self, z: int):
        self.canvas.set_zoom(z)
        try:
            self._ignore_zoom_bar = True
            self.zoom_bar.setValue(z)
        finally:
            self._ignore_zoom_bar = False

    def set_repeat(self, r: int):
        self.canvas.set_repeat(r)

    def set_image(self, image):
        self.image = image_to_qimage(image)
        self.canvas.set_image(self.image)

        self.info_label.setText(
            f"{self.image.width()}x{self.image.height()}"
        )

    def _zoom_bar_changed(self, value):
        if not self._ignore_zoom_bar:
            self.set_zoom(value)

    def _context_menu(self, pos: QPoint):
        menu = QMenu()
        if self.image is not None:
            menu.addAction(self.tr("Save image as ..."), self.save_image_as)

        menu.addSeparator()
        menu.addAction(self.tr("Load image ..."), self.load_image_dialog)
        for filename in self._last_loaded_images:
            menu.addAction(
                self.tr("Reload {filename}").format(Path(filename).name),
                partial(self.load_image, filename),
            )

        menu.exec(self.mapToGlobal(pos))

    def save_image_as(self):
        if self.image is not None:

            # copy the current image before opening dialog
            image = self.image.copy()

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

    def load_image_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption=self.tr("Load image"),
            filter="*.png",
        )
        if filename:
            self.load_image(filename)

            if filename in self._last_loaded_images:
                self._last_loaded_images.remove(filename)
            self._last_loaded_images.insert(0, filename)
            self._last_loaded_images = self._last_loaded_images[:4]

    def load_image(self, filename: Union[str, Path]):
        image = QImage(filename)
        self.set_image(image)


class ImageDisplayCanvas(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.image = None
        self._zoom = 100
        self.num_repeat = 1
        self._size = (10, 10)

    @property
    def zoom(self):
        return self._zoom

    def set_zoom(self, z):
        self._zoom = z
        self.setFixedSize(
            self._size[0] * self.zoom * self.num_repeat,
            self._size[1] * self.zoom * self.num_repeat)
        self.update()

    def set_repeat(self, r : int):
        self.num_repeat = max(1, r)
        self.setFixedSize(
            self._size[0] * self.zoom * self.num_repeat,
            self._size[1] * self.zoom * self.num_repeat)
        self.update()

    def set_image(self, img):
        self.image = img
        self._size = (self.image.width(), self.image.height())
        self.set_zoom(self.zoom)

    def paintEvent(self, e):
        if self.image is None:
            return
        p = QPainter(self)

        p.fillRect(
            0, 0,
            self.image.width() * self.num_repeat * self.zoom,
            self.image.height() * self.num_repeat * self.zoom,
            Qt.black
        )

        t = QTransform()
        t.scale(self.zoom / 100., self.zoom / 100.)

        p.setTransform(t)

        for y in range(self.num_repeat):
            for x in range(self.num_repeat):
                p.drawImage(x*self.image.width(), y*self.image.height(), self.image)
