from functools import partial

from .base import *
from ..tiling import LImageTiling
from ...dialogs import ParameterDialog


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
        {
            "name": "visibility",
            "type": "float",
            "default": .3,
            "min": 0.,
            "max": 1.,
        },
        {
            "name": "project_random_map",
            "type": "bool",
            "default": False,
        },
        {
            "name": "map_size",
            "type": "int2",
            "default": [8, 8],
            "min": [1, 1],
        },
        {
            "name": "map_seed",
            "type": "int",
            "default": 23,
            "min": 0,
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config["visibility"] = self.limage.ui_settings.tiling_visibility
        self.config["project_random_map"] = self.limage.ui_settings.project_random_tiling_map
        self.config["map_size"] = self.limage.ui_settings.tiling_map_size
        self.config["map_seed"] = self.limage.ui_settings.tiling_map_seed
        if self.limage.tiling is not None:
            self.config["tile_size"] = self.limage.tiling.tile_size
            self.config["offset"] = self.limage.tiling.offset

    def config_changed(self):
        self.limage.ui_settings.tiling_visibility = self.config["visibility"]
        self.limage.ui_settings.project_random_tiling_map = self.config["project_random_map"]
        self.limage.ui_settings.tiling_map_size = self.config["map_size"]
        self.limage.ui_settings.tiling_map_seed = self.config["map_seed"]
        if self.limage.tiling is None:
            self.limage.set_tiling(self._create_tiling())
        else:
            self.limage.tiling._tile_size = self.config["tile_size"]
            self.limage.tiling._offset = self.config["offset"]
        self.limage.ui_settings_changed()
        
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

        if self.limage.tiling and self.limage.tiling.attributes_map:
            menu.addAction("Repeat ...", self._repeat)
        menu.addSeparator()
        menu.addAction("Delete", self._delete)

    def _init_wang_tiling(self, mode: str, num_colors: int):
        self.limage.tiling.set_optimal_attributes_map(mode=mode, num_colors=num_colors)
        self.limage.set_tiling(self.limage.tiling)

    def _delete(self):
        self.limage.set_tiling(None)

    def _repeat(self):
        accepted, repeat_xy = ParameterDialog.get_parameter_value(
            {
                "name": "repeat_xy",
                "type": "int2",
                "default": [2, 2],
                "min": [1, 1],
            },
            title="Repeat image",
        )
        if not accepted:
            return

        tiling = self.limage.tiling
        size_x = max(k[0] for k in tiling.attributes_map.keys()) + 1
        size_y = max(k[1] for k in tiling.attributes_map.keys()) + 1

        for pos, attr in list(tiling.attributes_map.items()):
            for y in range(repeat_xy[1]):
                for x in range(repeat_xy[0]):
                    tiling.attributes_map[(pos[0] + x * size_x, pos[1] + y * size_y)] = deepcopy(attr)

        self.limage.set_tiling(tiling)

    def paint(self, painter: QPainter, event: QPaintEvent, canvas: "LImageCanvasWidget"):
        if not self.limage.ui_settings.project_random_tiling_map:
            if not canvas.is_visible("tiling_grid"):
                self.limage.paint_tiling_grid(painter)
            if not canvas.is_visible("tiling"):
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
