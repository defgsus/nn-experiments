import src.algo.sdf.two_d as sdf

import numpy as np

from .tiling import LImageTiling


class TilingTemplateRenderer:

    OBJECT_METHODS = (
        "edges_full", "edge_circles", "edge_triangles",
        "corners", "corner_triangles",
    )

    PARAMETERS = [
        {
            "name": "tile_resolution",
            "type": "int2",
            "default": [32, 32],
        },
        {
            "name": "template_size",
            "type": "int2",
            "default": [16, 16],
        },
        {
            "name": "render_color",
            "type": "int",
            "default": 0,
        },
        *(
            {
                "name": f"draw_{func_name}",
                "type": "bool",
                "default": False,
            }
            for func_name in OBJECT_METHODS
        ),
        {
            "name": "mask_radius",
            "type": "float",
            "default": .1,
        },
        {
            "name": "mask_offset",
            "type": "float",
            "default": 0.,
        },
    ]

    @classmethod
    def edges_full(
            cls,
            attr: LImageTiling.Attributes,
            color: int = 0,
            radius: float = 0.1,
    ):
        if all(c == color for c in attr.edge_colors):
            yield sdf.Box((2, 2))
        if attr.l == color:
            yield sdf.Box((1, radius)).translate((.5, 0))
        if attr.r == color:
            yield sdf.Box((1, radius)).translate((.5, 1))
        if attr.t == color:
            yield sdf.Box((radius, 1)).translate((0, .5))
        if attr.b == color:
            yield sdf.Box((radius, 1)).translate((1, .5))

    @classmethod
    def corners(
            cls,
            attr: LImageTiling.Attributes,
            color: int = 0,
            radius: float = 0.1,
    ):
        box = sdf.Circle(radius)
        if attr.tl == color:
            yield box
        if attr.tr == color:
            yield box.translate((0, 1))
        if attr.bl == color:
            yield box.translate((1, 0))
        if attr.br == color:
            yield box.translate((1, 1))

    @classmethod
    def edge_circles(
            cls,
            attr: LImageTiling.Attributes,
            color: int = 0,
            radius: float = 0.1,
    ):
        if attr.t == color and attr.l == color:
            yield sdf.Circle(1-radius).invert().translate((1, 1))
        if attr.t == color and attr.r == color:
            yield sdf.Circle(1-radius).invert().translate((1, 0))
        if attr.b == color and attr.l == color:
            yield sdf.Circle(1-radius).invert().translate((0, 1))
        if attr.b == color and attr.r == color:
            yield sdf.Circle(1-radius).invert().translate((0, 0))

    @classmethod
    def edge_triangles(
            cls,
            attr: LImageTiling.Attributes,
            color: int = 0,
            radius: float=.25,
            radius2: float = .5,
    ):
        o = sdf.Box(radius*np.sqrt(2)).rotate(45)
        io = sdf.Box((radius2, 2))
        if all(c == color for c in attr.edge_colors):
            yield sdf.Box(.5).translate(.5)
        if attr.l == color:
            yield o.intersect(io.rotate(90)).translate((.5, 0))
        if attr.r == color:
            yield o.intersect(io.rotate(90)).translate((.5, 1))
        if attr.t == color:
            yield o.intersect(io).translate((0, .5))
        if attr.b == color:
            yield o.intersect(io).translate((1, .5))

    @classmethod
    def corner_triangles(
            cls,
            attr: LImageTiling.Attributes,
            color: int = 0,
            radius: float=.25,
    ):
        o = sdf.Box(radius*np.sqrt(2)).rotate(45)
        if attr.tl == color:
            yield o.translate((0, 0))
        if attr.tr == color:
            yield o.translate((0, 1))
        if attr.bl == color:
            yield o.translate((1, 0))
        if attr.br == color:
            yield o.translate((1, 1))

    @classmethod
    def render(
            cls,
            tiling: LImageTiling,
            config: dict,
    ) -> np.ndarray:
        map_size = config["template_size"]
        tile_size = config["tile_resolution"]

        template = np.zeros((map_size[1] * tile_size[1], map_size[0] * tile_size[0], 3))

        for (x, y), attr in tiling.attributes_map.items():
            if 0 <= x < map_size[0] and 0 <= y < map_size[1]:
                tile = cls.render_tile(config=config, attr=attr)
                template[
                    y * tile_size[1]: (y+1) * tile_size[1],
                    x * tile_size[0]: (x+1) * tile_size[0],
                ] = tile

        return template

    @classmethod
    def render_tile(
            cls,
            config: dict,
            attr: LImageTiling.Attributes,
    ):
        tile_size = config["tile_resolution"]
        color = config["render_color"]

        objects = []
        for func_name in cls.OBJECT_METHODS:
            if config[f"draw_{func_name}"]:
                objects.extend(getattr(cls, func_name)(attr, color))

        objects = sdf.Union(*objects)

        mask = objects.render_mask(
            radius=config["mask_radius"],
            abs=False,
            shape=tile_size,
        )

        return mask[..., None].repeat(3, -1)



