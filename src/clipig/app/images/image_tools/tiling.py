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
            "$visible": "project_random_map",
        },
        {
            "name": "map_seed",
            "type": "int",
            "default": 23,
            "min": 0,
            "$visible": "project_random_map",
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._drag_color: Optional[int] = None
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
        sub_menu.addAction(
            f"1 color on edges+corners",
            partial(self._init_wang_tiling, "edge+corner", 1)
        )

        if self.limage.tiling and self.limage.tiling.attributes_map:
            menu.addAction("Shift ...", self._shift)
            menu.addAction("Repeat ...", self._repeat)
            menu.addAction("Crop ...", self._crop)
        menu.addSeparator()
        menu.addAction("Delete", self._delete)

        menu.addSeparator()
        menu.addAction(menu.tr("Render template ..."), self._render_template)

    def _init_wang_tiling(self, mode: str, num_colors: int):
        self.limage.tiling.set_optimal_attributes_map(mode=mode, num_colors=num_colors)
        self.limage.set_tiling(self.limage.tiling)

    def _delete(self):
        self.limage.set_tiling(None)

    def _render_template(self):
        from ..render_tiling_dialog import RenderTilingDialog
        layer = RenderTilingDialog.run_dialog(self.limage, self.project)
        if layer:
            template_layer = self.limage.get_layer(layer.name)
            if not template_layer:
                template_layer = self.limage.add_layer(layer.name)
            template_layer.set_image(layer.image)

    def _shift(self):
        accepted, (shift_x, shift_y) = ParameterDialog.get_parameter_value(
            {
                "name": "shift_xy",
                "type": "int2",
                "default": [0, 0],
                "min": [0, 0],
            },
            title="Shift tiling",
        )
        if not accepted:
            return

        tiling = self.limage.tiling
        size_x = max(k[0] for k in tiling.attributes_map.keys()) + 1
        size_y = max(k[1] for k in tiling.attributes_map.keys()) + 1

        old_attributes = deepcopy(tiling.attributes_map)
        tiling.attributes_map.clear()
        for (x, y), attr in list(old_attributes.items()):
            x2 = (x + shift_x) % size_x
            y2 = (y + shift_y) % size_y
            tiling.attributes_map[(x2, y2)] = attr

        self.limage.set_tiling(tiling)

    def _crop(self):
        tiling = self.limage.tiling
        size_x, size_y = tiling.map_size

        accepted, (crop_x, crop_y) = ParameterDialog.get_parameter_value(
            {
                "name": "crop_xy",
                "type": "int2",
                "default": [10, 10],
                "min": [1, 1],
            },
            title=f"Crop tiling ({size_x}x{size_y})",
        )
        if not accepted:
            return

        old_attributes = deepcopy(tiling.attributes_map)
        tiling.attributes_map.clear()
        for (x, y), attr in list(old_attributes.items()):
            if 0 <= x < crop_x and 0 <= y < crop_y:
                tiling.attributes_map[(x, y)] = attr

        self.limage.set_tiling(tiling)

    def _repeat(self):
        accepted, repeat_xy = ParameterDialog.get_parameter_value(
            {
                "name": "repeat_xy",
                "type": "int2",
                "default": [2, 2],
                "min": [1, 1],
            },
            title="Repeat tiling",
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
        if event.type in (MouseEvent.Drag, MouseEvent.Press) and event.button == Qt.MouseButton.LeftButton:
            if self.limage.tiling is None:
                tiling = self._create_tiling()
            else:
                tiling = self.limage.tiling

            tile_pos, attr_index = tiling.pixel_pos_to_tile_pos(event.pos, with_attribute_index=True)

            if tile_pos[0] >= 0 and tile_pos[1] >= 0:
                if self._drag_color is None:
                    color = tiling.get_tile_attribute(tile_pos, attr_index)
                    if color < 0 or color != self.config["tile_color"]:
                        self._drag_color = self.config["tile_color"]
                    else:
                        self._drag_color = -1

                tiling.set_tile_attribute(tile_pos, attr_index, self._drag_color)

                self.limage.set_tiling(tiling)

        if event.type == MouseEvent.Release:
            self._drag_color = None
