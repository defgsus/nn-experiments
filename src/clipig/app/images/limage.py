import dataclasses
import re
from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict

import PIL.Image
import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..util import image_to_qimage, qimage_to_torch, torch_to_qimage
from ..dialogs import ParameterDialog
from src.util.files import Filestream
from .tiling import LImageTiling


class LImageLayer:

    THUMBNAIL_SIZE = QSize(48, 48)

    def __init__(
            self,
            parent: "LImage",
            name: str,
            image: Optional[QImage] = None,
            repeat: Tuple[int, int] = (1, 1),
            active: bool = True,
            transparency: float = 0.,
            position: Tuple[int, int] = (0, 0),
    ):
        self._parent = parent
        self._name = name
        self._image = image
        self._repeat = tuple(repeat)
        self._thumbnail: Optional[QImage] = None
        self._active = active
        self._transparency = transparency
        self._position = tuple(position)

    def get_config(self) -> dict:
        return {
            "name": self._name,
            "active": self._active,
            "repeat": self._repeat,
            "transparency": self._transparency,
            "position": self._position,
        }

    def set_config(self, config: dict, emit: bool = True):
        self._name = config["name"]
        self._active = config["active"]
        self._repeat = config["repeat"]
        self._transparency = config.get("transparency", 0.)
        self._position = config.get("position", (0, 0))

        if emit:
            self.set_changed()

    def copy(self, parent: Optional["LImage"] = None) -> "LImageLayer":
        layer = LImageLayer(
            parent=self._parent if parent is None else parent,
            name=self._name,
            image=self._image.copy() if self._image is not None else None,
        )
        layer.set_config(self.get_config(), emit=False)
        return layer

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    def set_name(self, name: str):
        self._name = name
        self.set_changed()

    @property
    def image(self):
        return self._image

    def set_image(self, image: QImage):
        self._image = image
        self._thumbnail = None
        self.set_changed()

    @property
    def repeat(self):
        return self._repeat

    def set_repeat(self, repeat: Tuple[int, int]):
        self._repeat = (max(1, repeat[0]), max(1, repeat[1]))
        self._thumbnail = None
        self.set_changed()

    @property
    def position(self):
        return self._position

    def set_position(self, position: Tuple[int, int]):
        self._position = tuple(position)
        self._thumbnail = None
        self.set_changed()

    @property
    def active(self):
        return self._active

    def set_active(self, active: bool):
        self._active = active
        self.set_changed()

    @property
    def transparency(self):
        return self._transparency

    def set_transparency(self, transparency: float):
        self._transparency = transparency
        self.set_changed()

    @property
    def selected(self):
        return self._parent.selected_layer == self

    def set_selected(self):
        self._parent.set_selected_layer(self)

    def repeat_image(self, num_x: int = 1, num_y: int = 1):
        if num_x > 1 or num_y > 1 and self._image:
            rep_image = QImage(
                QSize(self._image.width() * num_x, self._image.height() * num_y),
                QImage.Format_ARGB32,
            )
            painter = QPainter(rep_image)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 100, 0, 255)))
            painter.drawRect(rep_image.rect())
            for y in range(num_y):
                for x in range(num_x):
                    painter.drawImage(x * self._image.width(), y * self._image.height(), self._image)
            painter.end()
            self.set_image(rep_image)

    def set_image_size(
            self,
            size: Union[QSize, Tuple[int, int]],
            transform_mode: Qt.TransformationMode = Qt.TransformationMode.SmoothTransformation,
            aspect_mode: Qt.AspectRatioMode = Qt.AspectRatioMode.IgnoreAspectRatio,
    ):
        if not isinstance(size, QSize):
            size = QSize(*size)

        if self._image is None:
            self._image = QImage(size, QImage.Format.Format_ARGB32)

        self._image = self._image.scaled(
            size,
            aspectRatioMode=aspect_mode,
            transformMode=transform_mode,
        )
        self.set_changed()

    def size(self) -> QSize:
        return self.rect().size()

    def content_rect(self) -> QRect:
        """
        The rect of the image (including repetitions)
        """
        if not self.image:
            return QRect()
        size = self.image.size()
        return QRect(0, 0, size.width() * self.repeat[0], size.height() * self.repeat[1])

    def rect(self) -> QRect:
        """
        The full rect of the layer starting at 0, 0
        """
        if not self.image:
            return QRect()
        rect = self.content_rect()
        rect.moveTo(*self.position)
        return QRect(0, 0, rect.right() + 1, rect.bottom() + 1)

    def to_torch(self) -> Optional[torch.Tensor]:
        if not self._image:
            return None
        return qimage_to_torch(self._image)

    def from_torch(self, image: torch.Tensor):
        self._image = torch_to_qimage(image)
        self.set_changed()

    def to_pil(self) -> Optional[PIL.Image.Image]:
        from torchvision.transforms.functional import to_pil_image
        if not self._image:
            return None
        return to_pil_image(qimage_to_torch(self._image))

    def to_clipboard(self):
        QApplication.clipboard().setImage(self.image)

    def paint(self, painter: QPainter):
        if self._image is None or not self._active:
            return

        painter.setOpacity(max(0., min(1., 1. - self.transparency)))

        for y in range(self._repeat[1]):
            for x in range(self._repeat[0]):
                painter.drawImage(
                    self.position[0] + x * self._image.width(),
                    self.position[1] + y * self._image.height(),
                    self._image,
                )

        painter.setOpacity(1.)

    def thumbnail(self) -> QImage():
        if not self._image:
            return QImage()

        if self._thumbnail is None:
            self._thumbnail = self._image.scaled(
                self.THUMBNAIL_SIZE,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.SmoothTransformation,
            )

        return self._thumbnail

    def set_changed(self):
        self._parent._layer_changed(self)

    def image_painter(self) -> "LImageLayerPainter":
        return LImageLayerPainter(self)


class LImageLayerPainter:

    def __init__(self, layer: LImageLayer):
        self.layer = layer
        self._painter: Optional[QPainter] = None

    def __enter__(self):
        if not self.layer.image:
            self._painter = QPainter()
        else:
            self._painter = QPainter(self.layer.image)

        return self._painter

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._painter:
            self._painter.end()
            # TODO: need to debounce this a little for performance, i think
            # self._thumbnail = None
            self.layer.set_changed()


class LImage:

    @dataclasses.dataclass
    class UISettings:
        tiling_visibility: float = .3
        project_random_tiling_map: bool = False
        tiling_map_size: Tuple[int, int] = (8, 8)
        tiling_map_seed: int = 23

    def __init__(self):
        from .limage_model import LImageModel

        self._layers: List[LImageLayer] = []
        self._selected_layer: Optional[LImageLayer] = None
        self._model: Optional[LImageModel] = None
        self._tiling: Optional[LImageTiling] = None
        self._tile_map: Optional[List[List[Tuple[int, int]]]] = None
        self._tiles: Optional[Dict[Tuple[int, int], QImage]] = None
        self._ui_settings = self.UISettings()

    @property
    def layers(self):
        return self._layers

    def is_empty(self) -> bool:
        return len(self.layers) == 0

    def rect(self, for_ui: bool = False) -> QRect:
        if not for_ui or not (self.ui_settings.project_random_tiling_map and self.tiling):
            full_rect = QRect()
            for layer in self.layers:
                if layer.active:
                    full_rect = full_rect.united(layer.rect())
            return full_rect

        return QRect(
            0, 0,
            self.tiling.tile_size[0] * self.ui_settings.tiling_map_size[0],
            self.tiling.tile_size[1] * self.ui_settings.tiling_map_size[1]
        )

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
        self._tiling = None
        self._layers_changed()

    def get_unique_layer_name(self, prefix: str):
        new_name = prefix
        count = 1
        while True:
            is_unique = True
            for layer in self._layers:
                if layer.name == new_name:
                    is_unique = False
                    break
            if is_unique:
                return new_name

            count += 1
            new_name = f"{prefix} #{count}"

    @property
    def tiling(self) -> Optional[LImageTiling]:
        return self._tiling

    def set_tiling(self, tiling: Optional[LImageTiling]):
        self._tiling = tiling
        self._layers_changed()

    @property
    def ui_settings(self):
        return self._ui_settings

    def ui_settings_changed(self):
        self._layers_changed()

    def add_layer(
            self,
            name: Optional[str] = None,
            image: Optional[QImage] = None,
            index: Optional[int] = None,
            repeat: Tuple[int, int] = (1, 1),
            active: bool = True,
            transparency: float = 0.,
            position: Tuple[int, int] = (0, 0),
    ) -> LImageLayer:
        if name is None:
            name = f"layer #{len(self.layers) + 1}"

        layer = LImageLayer(
            parent=self,
            name=name,
            image=image,
            repeat=repeat,
            active=active,
            transparency=transparency,
            position=position,
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

    def get_layer_torch_expression(self, expression: str) -> Optional[torch.Tensor]:
        """
        Supports things like:

            layer_name * .5 + other_layer_name
        """
        from src.util import calc

        _globals = {
            **vars(calc),
        }

        for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)", expression):
            layer_name = match.groups()[0]
            layer = self.get_layer(layer_name)
            if layer:
                #raise NameError(f"Layer '{layer_name}' not found for expression '{expression}'")
                _globals[layer_name] = layer.to_torch()

        try:
            return eval(expression, _globals)
        except Exception as e:
            raise RuntimeError(f"Expression '{expression}' failed with: {type(e).__name__}: {e}")

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

    def delete_layers(self, layers_or_names_or_indices: List[Union[LImageLayer, str, int]]):
        layers_to_remove = set(
            self.get_layer(i)
            for i in layers_or_names_or_indices
        )
        self._layers = [
            layer
            for layer in self._layers
            if layer not in layers_to_remove
        ]
        self._layers_changed()

    def duplicate_selected_layer(self) -> Optional[LImageLayer]:
        if self.selected_layer:
            layer = self.selected_layer.copy()
            self._layers.append(layer)
            self._selected_layer = layer
            self._layers_changed()
            return layer

    def paint(self, painter: QPainter, for_ui: bool = False):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not for_ui or not (self.ui_settings.project_random_tiling_map and self.tiling):
            for layer in self.layers:
                layer.paint(painter)

        else:
            self._update_tiles()
            s = self.tiling.tile_size
            for y, row in enumerate(self._tile_map):
                for x, tile_idx in enumerate(row):
                    if tile_idx in self._tiles:
                        painter.drawImage(x * s[0], y * s[1], self._tiles[tile_idx])

    def _update_tiles(self):
        if self._tiles is None:
            self._tiles = {}
            tile_template = self.to_qimage()
            s = self.tiling.tile_size
            for x, y in self.tiling.attributes_map.keys():
                if x * s[0] + s[0] <= tile_template.width() and y * s[1] + s[1] <= tile_template.height():
                    self._tiles[(x, y)] = tile_template.copy(x * s[0], y * s[1], s[0], s[1])
            self._tile_map = self.tiling.create_map_stochastic_scanline(
                size=self.ui_settings.tiling_map_size,
                seed=self.ui_settings.tiling_map_seed,
            )

    def pixel_pos_to_image_pos(self, positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not (self.ui_settings.project_random_tiling_map and self.tiling):
            return positions
        self._update_tiles()
        s = self.tiling.tile_size
        ret_pos = []
        for p in positions:
            # pixel pos
            x, y = p
            # map pos
            mx, my = x // s[0], y // s[1]
            if 0 <= mx < len(self._tile_map[0]) and 0 <= my < len(self._tile_map):
                # tile pos in image
                tx, ty = self._tile_map[my][mx]
                tx, ty = tx * s[0], ty * s[1]
                # tile pos + offset inside tile
                ret_pos.append((
                    tx + x - mx * s[0],
                    ty + y - my * s[1],
                ))
        return ret_pos

    def pixel_rects_to_image_rects(self, rects: List[QRect]) -> List[Tuple[QRect, Tuple[int, int]]]:
        """
        Project the rectangles into image space.

        This has only an effect when `tiling` is activated.

        Each rectangle will be moved to the original image position matching the mapped tile
        and possibly split into parts for each image rectangle.

        :param rects: list of QRect
        :return: list of tuples of (QRect, (int, int))
            First argument is the rectangle in image space,
            Second argument is the offset inside the provided rectangle
        """
        if not (self.ui_settings.project_random_tiling_map and self.tiling):
            return [(r, (0, 0)) for r in rects]

        self._update_tiles()
        s = self.tiling.tile_size
        ret_rects = []
        for rect in rects:
            mx, my = int(rect.x() // s[0]), int(rect.y() // s[1])
            mw, mh = int((rect.width() + s[0] - 1) // s[0] + 1), int((rect.height() + s[1] - 1) // s[1] + 1)
            # print(f"rect={rect}, mxy={(mx, my)}, mwh={(mw, mh)}")
            mx_, my_ = mx, my
            for my in range(my_, my_ + mh):
                for mx in range(mx_, mx_ + mw):
                    if 0 <= mx < len(self._tile_map[0]) and 0 <= my < len(self._tile_map):
                        tx, ty = self._tile_map[my][mx]
                        tx *= s[0]
                        ty *= s[1]
                        mrect: QRect = rect.__class__(mx * s[0], my * s[1], s[0], s[1])
                        irect = rect.intersected(mrect)
                        # print(f"INTESECT mx{mx}my{my} mrect={mrect}, irect={irect}")
                        if not irect.isEmpty():
                            image_rect = irect.__class__(irect)
                            image_rect.moveTo(
                                tx + irect.x() - mrect.x(),
                                ty + irect.y() - mrect.y(),
                            )
                            # print(f"rect={rect} mrect={mrect} irect={irect} image={image_rect}".replace("PyQt5.QtCore.QRect", ""))
                            ret_rects.append((
                                image_rect,
                                (irect.x() - rect.x(), irect.y() - rect.y())
                            ))
        return ret_rects

    def get_model(self):
        from .limage_model import LImageModel

        if self._model is None:
            self._model = LImageModel(None)
            self._model.set_limage(self)

        return self._model

    def _layer_changed(self, layer: LImageLayer):
        try:
            index = self.layers.index(layer)
        except ValueError:
            return

        self._tiles = None
        if self._model is not None:
            self._model.dataChanged.emit(
                self._model.index(index, 0),
                self._model.index(index, self._model.rowCount())
            )

    def _layers_changed(self):
        self._tiles = None
        if self._model is not None:
            self._model.modelReset.emit()

    def add_menu_actions(self, menu: QMenu, project: "ProjectWidget"):
        from .new_layer_dialog import NewLayerDialog
        from .image_transform_dialog import ImageTransformDialog
        menu.addAction(menu.tr("Add layer"), partial(NewLayerDialog.run_new_layer_dialog, self, menu))

        if self.selected_layer:
            sub_menu = QMenu(menu.tr("Layer") + f" \"{self.selected_layer.name}\"", menu)
            menu.addMenu(sub_menu)

            sub_menu.addAction(menu.tr("Copy to clipboard"), self.selected_layer.to_clipboard)

            sub_menu.addAction(menu.tr("Duplicate"), self.duplicate_selected_layer)
            sub_menu.addAction(
                menu.tr("Transform"),
                partial(ImageTransformDialog.run_dialog_on_limage_layer, limage=self, parent=menu, project=project),
            )

            def _repeat_action():
                accepted, repeat_xy = ParameterDialog.get_parameter_value(
                    {
                        "name": "repeat_xy",
                        "type": "int2",
                        "default": [2, 2],
                        "min": [1, 1],
                    },
                    title="Repeat image",
                    parent=menu,
                )
                if accepted:
                    self.selected_layer.repeat_image(*repeat_xy)
            sub_menu.addAction(menu.tr("Repeat ..."), _repeat_action)

    def to_qimage(self) -> QImage:
        image = QImage(self.size(), QImage.Format.Format_ARGB32)

        painter = QPainter(image)
        self.paint(painter)
        painter.end()

        return image

    def to_torch(self) -> torch.Tensor:
        return qimage_to_torch(self.to_qimage())

    def to_pil(self) -> PIL.Image.Image:
        from torchvision.transforms.functional import to_pil_image
        return to_pil_image(qimage_to_torch(self.to_qimage()))

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

    def copy_from(self, other: "LImage"):
        other = other.copy()
        self._layers = other._layers
        self._selected_layer = None if not self._layers else self._layers[0]
        for l in self._layers:
            l._parent = self
        self._layers_changed()

    def save_to_filestream(self, filestream: Filestream, directory: Union[str, Path]):
        directory = Path(directory)

        # -- build config file data --

        config_data = {
            "selected_index": self.selected_index,
            "ui_settings": vars(self._ui_settings),
            "tiling": None if self._tiling is None else self._tiling.get_config(),
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

        if config_data.get("tiling"):
            self._tiling = LImageTiling()
            self._tiling.set_config(config_data["tiling"])

        if config_data.get("ui_settings"):
            self._ui_settings = self.UISettings(**config_data["ui_settings"])

        if config_data["selected_index"] < len(self._layers):
            self._selected_layer = self._layers[config_data["selected_index"]]

    def paint_grid(
            self,
            painter: QPainter,
            size: Tuple[int, int],
            offset: Tuple[int, int] = (0, 0),
            visibility: float = .3,
    ):
        layer_size = self.size()

        painter.setCompositionMode(QPainter.CompositionMode_Xor)
        painter.setOpacity(visibility)
        painter.setPen(QPen(QColor(255, 255, 255)))
        for y in range(offset[1], layer_size.height(), size[1]):
            painter.drawLine(0, y, layer_size.width(), y)
        for x in range(offset[0], layer_size.width(), size[0]):
            painter.drawLine(x, 0, x, layer_size.height())

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setOpacity(1.)

    def paint_tiling_grid(self, painter: QPainter):
        if self._tiling is None:
            return

        self.paint_grid(
            painter, self._tiling.tile_size, self._tiling.offset, visibility=self.ui_settings.tiling_visibility,
        )

    def paint_tiling(self, painter: QPainter):
        if self._tiling is None:
            return

        tile_size = self._tiling.tile_size
        tiling_offset = QPoint(*self._tiling.offset)
        visibility = self.ui_settings.tiling_visibility

        # self.paint_grid(painter, tile_size, self._tiling.offset)

        TILING_COLORS = [
            QColor(255, 128, 96),
            QColor(96, 255, 128),
            QColor(96, 128, 255),
            QColor(128, 255, 224),
        ]

        for (x, y), attr in self._tiling.attributes_map.items():
            offset = (
                    QPoint(x * self._tiling.tile_size[0], y * self._tiling.tile_size[1])
                    + tiling_offset
            )
            for pos_idx, color_idx in enumerate(attr.colors):
                if color_idx >= 0:
                    poly = self._tiling.get_tile_polygon(pos_idx, offset)
                    color = TILING_COLORS[color_idx % len(TILING_COLORS)]

                    painter.setOpacity(visibility)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(color))
                    painter.drawPolygon(poly)

                    painter.setOpacity(min(1., visibility * 1.3))
                    painter.setPen(QPen(color.darker(200)))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolygon(poly)

        painter.setOpacity(1)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
