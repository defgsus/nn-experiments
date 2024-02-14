from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Tuple

import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..util import image_to_qimage, qimage_to_torch
from src.util.files import Filestream


class LImageLayer:

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
        self._parent.set_selected_layer(self)

    def size(self) -> QSize:
        return self.rect().size()

    def rect(self) -> QRect:
        if not self.image:
            return QRect()
        rect = self.content_rect()
        return QRect(0, 0, rect.right(), rect.bottom())

    def content_rect(self) -> QRect:
        if not self.image:
            return QRect()
        rect = self.image.rect()
        return QRect(rect.left(), rect.top(), rect.width() * self.repeat[0], rect.height() * self.repeat[1])

    def copy(self, parent: Optional["LImage"] = None) -> "LImageLayer":
        return LImageLayer(
            parent=self._parent if parent is None else parent,
            name=self._name,
            image=self._image.copy() if self._image is not None else None,
            repeat=self._repeat,
            active=self._active,
        )

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

    def get_config(self) -> dict:
        return {
            "name": self._name,
            "active": self._active,
            "repeat": self._repeat,
        }

    def set_config(self, config: dict):
        self._name = config["name"]
        self._active = config["active"]
        self._repeat = config["repeat"]
        self._parent._layer_changed(self)


class LImage:

    def __init__(self):
        from .limage_model import LImageModel

        self._layers: List[LImageLayer] = []
        self._selected_layer: Optional[LImageLayer] = None
        self._model: Optional[LImageModel] = None

    @property
    def layers(self):
        return self._layers

    def is_empty(self) -> bool:
        return len(self.layers) == 0

    def rect(self) -> QRect:
        full_rect = QRect()
        for layer in self.layers:
            if layer.active:
                full_rect = full_rect.united(layer.rect())
        return full_rect

    def content_rect(self) -> QRect:
        full_rect = QRect()
        for layer in self.layers:
            if layer.active:
                full_rect = full_rect.united(layer.content_rect())
        return full_rect

    def size(self) -> QSize:
        return self.rect().size()

    def clear(self):
        self.layers.clear()
        self._layers_changed()

    def add_layer(
            self,
            name: Optional[str] = None,
            image: Optional[str] = None,
            index: Optional[int] = None,
    ) -> LImageLayer:
        if name is None:
            name = f"layer #{len(self.layers) + 1}"

        layer = LImageLayer(
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

    def get_layer(self, layer_or_name_or_index: Union[LImageLayer, str, int]) -> Optional[LImageLayer]:
        if isinstance(layer_or_name_or_index, LImageLayer):
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
    def selected_layer(self) -> Optional[LImageLayer]:
        return self._selected_layer

    @property
    def selected_index(self) -> int:
        if self._selected_layer is not None:
            for i, l in enumerate(self._layers):
                if l == self._selected_layer:
                    return i
        return -1

    def set_selected_layer(self, layer_or_name_or_index: Union[LImageLayer, str, int]):
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

    def _layer_changed(self, layer: LImageLayer):
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

    def add_menu_actions(self, menu: QMenu):
        menu.addAction("Add layer", lambda : self.add_layer())

    def to_qimage(self) -> QImage:
        image = QImage(self.size(), QImage.Format.Format_ARGB32)

        painter = QPainter(image)
        self.paint(painter)
        painter.end()

        return image

    def to_torch(self) -> torch.Tensor:
        return qimage_to_torch(self.to_qimage())

    def copy(self, merged: bool = False) -> "LImage":
        if merged:
            limage = LImage()
            limage.add_layer(image=self.to_qimage())

        else:
            limage = LImage()
            limage._layers = [
                layer.copy(parent=limage)
                for layer in self.layers
            ]
            if limage._layers:
                limage._selected_layer = limage._layers[0]

        return limage

    def save_to_filestream(self, filestream: Filestream, directory: Union[str, Path]):
        directory = Path(directory)

        # -- build config file data --

        config_data = {
            "selected_index": self.selected_index,
            "layers": [],
        }

        for idx, layer in enumerate(self.layers):
            layer: LImageLayer

            layer_filename_part = f"layer_{idx:02}"

            layer_config = {
                "config": layer.get_config(),
            }
            if layer.image is not None:
                image_filename = str(directory / f"{layer_filename_part}.png")
                layer_config["image_filename"] = image_filename

            config_data["layers"].append(layer_config)

        filestream.write_yaml(directory / f"config.yaml", config_data)

        # -- write images --

        for layer_config, layer in zip(config_data["layers"], self._layers):
            if layer_config.get("image_filename"):
                filestream.write_qimage(layer_config["image_filename"], layer.image)

    def load_from_filestream(self, filestream: Filestream, config_filename: Union[str, Path]):
        config_data = filestream.read_yaml(config_filename)

        self.clear()

        for layer_config in config_data["layers"]:
            layer = self.add_layer()

            if layer_config.get("image_filename"):
                layer._image = filestream.read_qimage(layer_config["image_filename"])

            layer.set_config(layer_config["config"])
