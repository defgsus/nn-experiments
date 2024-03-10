from functools import partial

from .base import *


class SelectionTool(ImageToolBase):

    NAME = "select"


class MoveTool(ImageToolBase):

    NAME = "move"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_pos = None
        self._mouse_pos = None
        
    def mouse_event(self, event: MouseEvent):
        if event.type == MouseEvent.Press:
            self._layer_pos = self.layer.position
            self._mouse_pos = event.pos

        if event.type == MouseEvent.Drag and self._layer_pos:
            self.layer.set_position((
                self._layer_pos[0] + event.x - self._mouse_pos[0],
                self._layer_pos[1] + event.y - self._mouse_pos[1],
            ))


class TilingTool(ImageToolBase):

    NAME = "tiling"

    PARAMS = [
        {
            "name": "tile_size",
            "type": "int2",
            "default": [32, 32],
            "min": [1, 1],
        },
        {
            "name": "offset",
            "type": "int2",
            "default": [0, 0],
            "min": [0, 0],
        },
        {
            "name": "tile_color",
            "type": "int",
            "default": 0,
            "min": 0,
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..tiling import LImageTiling

        if self.layer.tiling is not None:
            self.config["tile_size"] = self.layer.tiling.tile_size
            self.config["offset"] = self.layer.tiling.offset

    def config_changed(self):
        self.layer.tiling._tile_size = self.config["tile_size"]
        self.layer.tiling._offset = self.config["offset"]

    def add_menu_actions(self, menu: QMenu, project: "ProjectWidget"):

        sub_menu = QMenu("Optimal wang tiling", menu)
        menu.addMenu(sub_menu)
        for mode in ("edge", "corner"):
            for num_colors in (2, 3):
                sub_menu.addAction(
                    f"Initialize with {num_colors} {mode} colors",
                    partial(self._init_wang_tiling, mode, num_colors)
                )

        menu.addAction("Delete", self._delete)

    def _init_wang_tiling(self, mode: str, num_colors: int):
        self._tiling.set_optimal_attributes_map(mode=mode, num_colors=num_colors)
        self.layer.set_tiling(self._tiling)

    def _delete(self):
        self._tiling.clear_attributes()
        self.layer.set_tiling(None)

    def XXX_paint(self, painter: QPainter, event: QPaintEvent):
        layer_size = self.layer.size()
        tile_size = self.config["tile_size"]
        offset = self.config["tile_size"]

        painter.setCompositionMode(QPainter.CompositionMode_Xor)
        painter.setOpacity(.3)
        painter.setPen(QPen(QColor(255, 255, 255)))
        for y in range(offset[1], layer_size.height(), tile_size[1]):
            painter.drawLine(0, y, layer_size.width(), y)
        for x in range(offset[0], layer_size.width(), tile_size[0]):
            painter.drawLine(x, 0, x, layer_size.height())

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        tiling_offset = QPoint(*self._tiling.offset)
        for (x, y), attr in self._tiling.attributes_map.items():
            offset = (
                QPoint(x * self._tiling.tile_size[0], y * self._tiling.tile_size[1])
                + tiling_offset
            )
            for pos_idx, color_idx in enumerate(attr.colors):
                if color_idx >= 0:
                    poly = self._tiling.get_tile_polygon(pos_idx, offset)
                    color = self.TILING_COLORS[color_idx % len(self.TILING_COLORS)]

                    painter.setOpacity(.3)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(color))
                    painter.drawPolygon(poly)

                    painter.setOpacity(.5)
                    painter.setPen(QPen(color.darker(200)))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolygon(poly)

        painter.setOpacity(1)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

    def mouse_event(self, event: MouseEvent):
        if event.type == MouseEvent.Press and event.button == Qt.MouseButton.LeftButton:
            tile_pos, attr_index = self._tiling.pixel_pos_to_tile_pos(event.pos, with_attribute_index=True)

            color = self._tiling.get_tile_attribute(tile_pos, attr_index)
            if color < 0 or color != self.config["tile_color"]:
                color = self.config["tile_color"]
            else:
                color = -1

            self._tiling.set_tile_attribute(tile_pos, attr_index, color)

            self.layer.set_tiling(self._tiling)
