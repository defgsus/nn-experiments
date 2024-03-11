from functools import partial

from .base import *
from ..tiling import LImageTiling


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

        if self.limage.tiling is not None:
            self.config["tile_size"] = self.limage.tiling.tile_size
            self.config["offset"] = self.limage.tiling.offset

    def config_changed(self):
        if self.limage.tiling is None:
            self.limage.set_tiling(self._create_tiling())
        else:
            self.limage.tiling._tile_size = self.config["tile_size"]
            self.limage.tiling._offset = self.config["offset"]

    def _create_tiling(self):
        return LImageTiling(
            tile_size=self.config["tile_size"],
            offset=self.config["offset"],
        )

    def add_menu_actions(self, menu: QMenu, project: "ProjectWidget"):
        sub_menu = QMenu("Initialize with optimal wang tiling", menu)
        menu.addMenu(sub_menu)
        for num_colors in (2, 3):
            for mode in ("edge", "corner"):
                sub_menu.addAction(
                    f"{num_colors} colors on {mode}s",
                    partial(self._init_wang_tiling, mode, num_colors)
                )

        menu.addAction("Delete", self._delete)

    def _init_wang_tiling(self, mode: str, num_colors: int):
        self.limage.tiling.set_optimal_attributes_map(mode=mode, num_colors=num_colors)
        self.limage.set_tiling(self.limage.tiling)

    def _delete(self):
        self.limage.set_tiling(None)

    def paint(self, painter: QPainter, event: QPaintEvent):
        self.limage.paint_tiling(painter)

    def mouse_event(self, event: MouseEvent):
        if event.type == MouseEvent.Press and event.button == Qt.MouseButton.LeftButton:
            if self.limage.tiling is None:
                tiling = self._create_tiling()
            else:
                tiling = self.limage.tiling

            tile_pos, attr_index = tiling.pixel_pos_to_tile_pos(event.pos, with_attribute_index=True)

            color = tiling.get_tile_attribute(tile_pos, attr_index)
            if color < 0 or color != self.config["tile_color"]:
                color = self.config["tile_color"]
            else:
                color = -1

            tiling.set_tile_attribute(tile_pos, attr_index, color)

            self.limage.set_tiling(tiling)
