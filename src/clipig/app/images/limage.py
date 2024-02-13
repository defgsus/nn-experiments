from functools import partial
from pathlib import Path
import dataclasses
from typing import Optional, List, Union, Tuple

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..util import image_to_qimage


class ImageLayer:

    THUMBNAIL_SIZE = QSize(48, 48)

    def __init__(
            self,
            parent: "LImage",
            name: str,
            image: Optional[QImage] = None,
            repeat: Tuple[int, int] = (1, 1),
            active: bool = True,
    ):
        self._parent = parent
        self._name = name
        self._image = image
        self._repeat = repeat
        self._thumbnail: Optional[QImage] = None
        self._active = active

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    @property
    def image(self):
        return self._image

    @property
    def repeat(self):
        return self._repeat

    @property
    def active(self):
        return self._active

    def set_active(self, active: bool):
        self._active = active
        self._parent._layer_changed(self)

    @property
    def selected(self):
        return self._parent.selected_layer == self

    def set_selected(self):
        self.parent.set_selected_layer(self)

    def size(self) -> QSize:
        return self.rect().size()

    def rect(self) -> QRect:
        if not self.image:
            return QRect()
        rect = self.image.rect()
        return QRect(rect.left(), rect.top(), rect.width() * self.repeat[0], rect.height() * self.repeat[1])

    def paint(self, painter: QPainter):
        if self._image is None or not self._active:
            return

        for y in range(self._repeat[1]):
            for x in range(self._repeat[0]):
                painter.drawImage(
                    x * self._image.width(),
                    y * self._image.height(),
                    self._image,
                )

    def set_image(self, image: QImage):
        self._image = image
        self._thumbnail = None
        self._parent._layer_changed(self)

    def thumbnail(self) -> QImage():
        if not self._image:
            return QImage()

        if self._thumbnail is None:
            self._thumbnail = self._image.scaled(
                self.THUMBNAIL_SIZE,
                aspectRatioMode=Qt.KeepAspectRatio,
                transformMode=Qt.SmoothTransformation,
            )

        return self._thumbnail


class LImage:

    def __init__(self):
        from .limage_model import LImageModel

        self.layers: List[ImageLayer] = []
        self._selected_layer: Optional[ImageLayer] = None
        self._model: Optional[LImageModel] = None

    def rect(self) -> QRect:
        full_rect = QRect()
        for layer in self.layers:
            if layer.active:
                full_rect = full_rect.united(layer.rect())
        return full_rect

    def size(self) -> QSize():
        return self.rect().size()

    def add_layer(
            self,
            name: Optional[str] = None,
            image: Optional[str] = None,
            index: Optional[int] = None,
    ) -> ImageLayer:
        if name is None:
            name = f"layer #{len(self.layers) + 1}"

        layer = ImageLayer(
            parent=self,
            name=name,
            image=image,
        )

        if index is not None:
            self.layers.insert(index, layer)
        else:
            self.layers.append(layer)

        self._selected_layer = layer
        self._layers_changed()

        return layer

    def get_layer(self, layer_or_name_or_index: Union[ImageLayer, str, int]) -> Optional[ImageLayer]:
        if isinstance(layer_or_name_or_index, ImageLayer):
            if layer_or_name_or_index.parent == self:
                return layer_or_name_or_index

        elif isinstance(layer_or_name_or_index, int):
            if 0 <= layer_or_name_or_index < len(self.layers):
                return self.layers[layer_or_name_or_index]

        elif isinstance(layer_or_name_or_index, str):
            for layer in self.layers:
                if layer.name == layer_or_name_or_index:
                    return layer

    @property
    def selected_layer(self) -> Optional[ImageLayer]:
        return self._selected_layer

    def set_selected_layer(self, layer_or_name_or_index: Union[ImageLayer, str, int]):
        layer = self.get_layer(layer_or_name_or_index)
        if not layer:
            return

        changed = layer != self._selected_layer
        self._selected_layer = layer

        if changed:
            self._layers_changed()

    def paint(self, painter: QPainter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for layer in self.layers:
            layer.paint(painter)

    def get_model(self):
        from .limage_model import LImageModel

        if self._model is None:
            self._model = LImageModel(None)
            self._model.set_limage(self)

        return self._model

    def _layer_changed(self, layer: ImageLayer):
        index = self.layers.index(layer)
        if index < 0:
            return

        if self._model is not None:
            self._model.dataChanged.emit(
                self._model.index(index, 0),
                self._model.index(index, self._model.rowCount())
            )

    def _layers_changed(self):
        if self._model is not None:
            self._model.modelReset.emit()
