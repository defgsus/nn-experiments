import dataclasses
import math
import random
from copy import deepcopy
from typing import Optional, List, Union, Tuple, Dict, Iterable

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import torch
import numpy as np

from src.util.image import get_image_window


class LImageTiling:

    class PosIndex:
        T = 0
        TL = 1
        L = 2
        BL = 3
        B = 4
        BR = 5
        R = 6
        TR = 7

    @dataclasses.dataclass
    class Attributes:
        """
        Edge and corner colors of a tile. `-1` is undefined.
        """
        t: int = -1
        tl: int = -1
        l: int = -1
        bl: int = -1
        b: int = -1
        br: int = -1
        r: int = -1
        tr: int = -1

        def __hash__(self):
            return self.colors

        def __repr__(self):
            args = ", ".join(f"{key}: {value}" for key, value in vars(self).items() if value != -1)
            return f"{self.__class__.__name__}({args})"

        @property
        def colors(self) -> Tuple[int, ...]:
            return self.t, self.tl, self.l, self.bl, self.b, self.br, self.r, self.tr

        def has_attributes(self) -> bool:
            return (
                self.t >= 0 or self.tl >= 0 or self.l >= 0 or self.bl >= 0
                or self.b >= 0 or self.br >= 0 or self.r >= 0 or self.tr >= 0
            )

        def num_elements(self) -> int:
            num = 0
            for c in self.colors:
                if c >= 0:
                    num += 1
            return num

        def set_color(self, index: int, color: int):
            if index == LImageTiling.PosIndex.T:
                self.t = color
            elif index == LImageTiling.PosIndex.TL:
                self.tl = color
            elif index == LImageTiling.PosIndex.L:
                self.l = color
            elif index == LImageTiling.PosIndex.BL:
                self.bl = color
            elif index == LImageTiling.PosIndex.B:
                self.b = color
            elif index == LImageTiling.PosIndex.BR:
                self.br = color
            elif index == LImageTiling.PosIndex.R:
                self.r = color
            elif index == LImageTiling.PosIndex.TR:
                self.tr = color

        def matches_left(self, other: "LImageTiling.Attributes") -> bool:
            if self.tl != other.tr:
                return False
            if self.l != other.r:
                return False
            if self.bl != other.br:
                return False
            if self.tl >= 0 and self.tl != other.tr:
                return False
            if self.l >= 0 and self.l != other.r:
                return False
            if self.bl >= 0 and self.bl != other.br:
                return False
            return True

        def matches_right(self, other: "LImageTiling.Attributes") -> bool:
            return other.matches_left(self)

        def matches_top(self, other: "LImageTiling.Attributes") -> bool:
            if self.tl != other.bl:
                return False
            if self.t != other.b:
                return False
            if self.tr != other.br:
                return False
            if self.tl != other.bl:
                return False
            if self.t >= 0 and self.t != other.b:
                return False
            if self.tr >= 0 and self.tr != other.br:
                return False
            return True

        def matches_bottom(self, other: "LImageTiling.Attributes") -> bool:
            return other.matches_top(self)

    def __init__(
            self,
            tile_size: Union[Iterable[int], Tuple[int, int], QPoint, QSize] = (32, 32),
            offset: Union[Iterable[int], Tuple[int, int], QPoint] = (0, 0),
    ):
        self._tile_size = self._pos_to_tuple(tile_size)
        self._offset = self._pos_to_tuple(offset)
        self._attributes_map: Dict[Tuple[int, int], LImageTiling.Attributes] = {}

    def get_config(self) -> dict:
        return {
            "tile_size": self._tile_size,
            "offset": self._offset,
            "attributes_map": {
                f"{x},{y}": vars(attr)
                for (x, y), attr in self._attributes_map.items()
            }
        }

    def set_config(self, config: dict):
        config = deepcopy(config)
        self._tile_size = config["tile_size"]
        self._offset = config["offset"]
        self._attributes_map = {
            tuple(int(i) for i in key.split(",")): self.Attributes(**value)
            for key, value in config["attributes_map"].items()
        }

    def copy(self):
        tiling = self.__class__()
        tiling.set_config(self.get_config())
        return tiling

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def offset(self):
        return self._offset

    @property
    def attributes_map(self) -> Dict[Tuple[int, int], "LImageTiling.Attributes"]:
        return self._attributes_map

    def set_attributes_map(self, attributes_map: Dict[Tuple[int, int], "LImageTiling.Attributes"]):
        self._attributes_map = deepcopy(attributes_map)

    def set_optimal_attributes_map(self, num_colors: int = 2, mode: str = "edge"):
        from src.algo import wangtiles
        opt_indices = wangtiles.OPTIMAL_WANG_INDICES_SQUARE.get((mode, num_colors))
        if opt_indices is None:
            raise NotImplementedError(f"Sorry, no optimal tile indices for {num_colors}-{mode}")
        wang_tiles = wangtiles.WangTiles(colors=wangtiles.get_wang_tile_colors(num_colors), mode=mode)

        self._attributes_map = {}
        for y, row in enumerate(opt_indices):
            for x, idx in enumerate(row):
                t = wang_tiles.tiles[idx]
                self._attributes_map[(x, y)] = self.Attributes(
                    t=t.top, tl=t.top_left, l=t.left, bl=t.bottom_left,
                    b=t.bottom, br=t.bottom_right, r=t.right, tr=t.top_right,
                )

    def tile_attributes(self, pos: Union[Iterable[int], Tuple[int, int], QPoint]) -> Attributes:
        return self._attributes_map.get(self._pos_to_tuple(pos)) or self.Attributes()

    def get_tile_attribute(
            self,
            pos: Union[Iterable[int], Tuple[int, int], QPoint],
            attr_index: int,
    ) -> int:
        pos = self._pos_to_tuple(pos)
        if pos not in self._attributes_map:
            return -1
        return self._attributes_map[pos].colors[attr_index]

    def set_tile_attribute(
            self,
            pos: Union[Iterable[int], Tuple[int, int], QPoint],
            attr_index: int,
            color: int,
    ):
        pos = self._pos_to_tuple(pos)
        if pos not in self._attributes_map:
            if color < 0:
                return
            self._attributes_map[pos] = self.Attributes()
            self._attributes_map[pos].set_color(attr_index, color)
        else:
            self._attributes_map[pos].set_color(attr_index, color)
            # TODO: keep them in to allow empty tiles to render
            #if not self._attributes_map[pos].has_attributes():
            #    del self._attributes_map[pos]

    def clear_attributes(self):
        self._attributes_map.clear()

    def _pos_to_tuple(self, pos: Union[Iterable[int], Tuple[int, int], QPoint]) -> Tuple[int, int]:
        if isinstance(pos, QPoint):
            return pos.x(), pos.y()
        elif isinstance(pos, QSize):
            return pos.width(), pos.height()
        else:
            pos = tuple(pos)
            if len(pos) != 2:
                raise ValueError(f"Expected 2 values, got {pos}")
            return pos

    def get_tile_polygon(self, pos_index: int, offset: Optional[Union[Tuple[int, int], QPoint]] = None):
        thickness = .27
        thickness2 = .5
        w, h = self.tile_size
        wo, ho = int(w * thickness), int(h * thickness)
        wo2, ho2 = int(w * thickness2), int(h * thickness2)
        if pos_index == self.PosIndex.T:
            points = [(0, 0), (w, 0), (w - wo, ho), (wo, ho)]
        elif pos_index == self.PosIndex.TL:
            points = [(0, 0), (wo2, 0), (0, ho2)]
        elif pos_index == self.PosIndex.L:
            points = [(0, 0), (wo, ho), (wo, h - ho), (0, h)]
        elif pos_index == self.PosIndex.BL:
            points = [(0, h), (0, h - ho2), (wo2, h)]
        elif pos_index == self.PosIndex.B:
            points = [(0, h), (wo, h - ho), (w - wo, h - ho), (w, h)]
        elif pos_index == self.PosIndex.BR:
            points = [(w, h), (w - wo2, h), (w, h - ho2)]
        elif pos_index == self.PosIndex.R:
            points = [(w, 0), (w, h), (w - wo, h - ho), (w - wo, ho)]
        elif pos_index == self.PosIndex.TR:
            points = [(w, 0), (w, ho2), (w - wo2, 0)]
        else:
            raise ValueError(f"Expected `pos_index` in range [0, 7], got {pos_index}")

        if offset is not None:
            offset = self._pos_to_tuple(offset)
            points = [QPoint(x + offset[0], y + offset[1]) for x, y in points]
        else:
            points = [QPoint(x, y) for x, y in points]

        return QPolygon(points)

    def pixel_pos_to_tile_pos(
            self,
            pos: Union[Iterable[int], Tuple[int, int], QPoint],
            with_attribute_index: bool = False,
    ) -> Union[Tuple[int, int], Tuple[Tuple[int, int], int]]:
        pos = self._pos_to_tuple(pos)
        tile_pos_f = (
            (pos[0] - self._offset[0]) / self._tile_size[0],
            (pos[1] - self._offset[1]) / self._tile_size[1],
        )
        tile_pos = tuple(int(math.floor(i)) for i in tile_pos_f)

        if not with_attribute_index:
            return tile_pos

        tile_pos_f = tuple(i - math.floor(i) for i in tile_pos_f)

        attr_index = -1
        thick = .333
        if tile_pos_f[0] < thick:
            if tile_pos_f[1] < thick:
                attr_index = self.PosIndex.TL
            elif tile_pos_f[1] >= 1. - thick:
                attr_index = self.PosIndex.BL
            else:
                attr_index = self.PosIndex.L

        elif tile_pos_f[0] >= 1. - thick:
            if tile_pos_f[1] < thick:
                attr_index = self.PosIndex.TR
            elif tile_pos_f[1] >= 1. - thick:
                attr_index = self.PosIndex.BR
            else:
                attr_index = self.PosIndex.R

        else:
            if tile_pos_f[1] < thick:
                attr_index = self.PosIndex.T
            elif tile_pos_f[1] >= 1. - thick:
                attr_index = self.PosIndex.B

        return tile_pos, attr_index

    def create_map_stochastic_scanline(
            self,
            size: Tuple[int, int],
            seed: Optional[int] = None,
    ):
        rng = random.Random(seed)

        possible_tiles = list(self.attributes_map.keys())

        map = [[(0, 0)] * size[0] for _ in range(size[1])]
        for y in range(size[1]):
            for x in range(size[0]):

                rng.shuffle(possible_tiles)
                for tile_idx in possible_tiles:
                    attrs = self.tile_attributes(tile_idx)

                    if y > 0 and not attrs.matches_top(self.tile_attributes(map[y - 1][x])):
                        continue
                    if x > 0 and not attrs.matches_left(self.tile_attributes(map[y][x - 1])):
                        continue

                    map[y][x] = tile_idx
                    break

        return map

    def create_map_stochastic_perlin(
            self,
            size: Tuple[int, int],
            seed: Optional[int] = None,
    ):
        from src.algo.sdf.two_d.util import perlin_noise_2d
        rng = random.Random(seed)
        tiles_outside = []
        tiles_inside = []
        for idx, a in self.attributes_map.items():
            n = a.num_elements()
            if n == 0:
                tiles_outside.append(idx)
            if n == 4:
                tiles_inside.append(idx)
        noise = perlin_noise_2d(
            shape=(size[1], size[0]),
            res=(size[1]//3, size[0]//3),
            rng=np.random.RandomState(seed),
        )
        map = np.tile(np.array([[None]], dtype=np.object_), noise.shape)
        for y, row in enumerate(map):
            for x, v in enumerate(row):
                if noise[y, x] > .1:
                    map[y, x] = repr(rng.choice(tiles_inside))
                if noise[y, x] < -.3:
                    map[y, x] = repr(rng.choice(tiles_outside))

        all_tiles = list(self.attributes_map.keys())
        for i in range(size[0]*size[1] // 20):
            x, y = rng.randrange(size[0]), rng.randrange(size[1])
            if map[y, x] is None or rng.uniform(0, 1) < .1:
                map[y, x] = repr(rng.choice(all_tiles))
        map = [
            [eval(x) if x else None for x in row]
            for row in map
        ]
        self.set_map_default(map, (0, 0))
        for i in range(10):
            has_errors, is_solved = self.resolve_map(map)
            if is_solved:
                break
        return map

    def set_map_default(self, map: List[List[Optional[Tuple[int, int]]]], default: Tuple[int, int] = (0, 0)):
        for row in map:
            for x, v in enumerate(row):
                if v is None:
                    row[x] = default

    def tile_matches_map(
            self,
            map: List[List[Optional[Tuple[int, int]]]],
            attrs: "LImageTiling.Attributes",
            x: int,
            y: int,
    ) -> bool:
        if y > 0 and map[y-1][x] is not None and not attrs.matches_top(self.tile_attributes(map[y - 1][x])):
            return False
        if y < len(map)-1 and map[y+1][x] is not None and not attrs.matches_bottom(self.tile_attributes(map[y + 1][x])):
            return False
        if x > 0 and map[y][x-1] is not None and not attrs.matches_left(self.tile_attributes(map[y][x - 1])):
            return False
        if x < len(map[0])-1 and map[y][x+1] is not None and not attrs.matches_right(self.tile_attributes(map[y][x + 1])):
            return False
        return True

    def resolve_map(
            self,
            map: List[List[Optional[Tuple[int, int]]]],
    ):
        def _iter_mismatches():
            positions = [(x, y) for y in range(len(map)) for x in range(len(map[0]))]
            random.shuffle(positions)
            for x, y in positions:
                #for y in range(len(map)):
                #    for x in range(len(map[0])):
                if map[y][x] is not None and not self.tile_matches_map(map, map[y][x], x, y):
                    yield x, y

        def _find_tile(x, y):
            for tile_idx in self.attributes_map.keys():
                if self.tile_matches_map(map, tile_idx, x, y):
                    return tile_idx

        all_tiles = list(self.attributes_map.keys())
        has_errors = False
        is_solved = True
        for x, y in _iter_mismatches():
            has_errors = True
            t = _find_tile(x, y)
            if t is None:
                is_solved = False
                #t = random.choice(all_tiles)

            map[y][x] = t

        return has_errors, is_solved

    def render_tile_map(
            self,
            template: torch.Tensor,
            map: List[List[Tuple[int, int]]],
            overlap: Union[int, Tuple[int, int]] = 0,
            default_tile_idx: Tuple[int, int] = (0, 0)
    ):
        if isinstance(overlap, int):
            overlap = (overlap, overlap)

        if overlap[-2] > self.tile_size[-2] or overlap[-1] > self.tile_size[-1]:
            raise ValueError(
                f"`overlap` exceeds tile size, got {overlap}, tile shape is {self.tile_size}"
            )

        image = torch.zeros(
            template.shape[-3],
            len(map[0]) * self.tile_size[1],
            len(map[1]) * self.tile_size[0],
            dtype=template.dtype,
            device=template.device,
        )
        if overlap != (0, 0):
            accum = torch.zeros_like(image)
            window = get_image_window(list(reversed(self.tile_size))).to(template)

        for y, row in enumerate(map):
            for x, tile_pos in enumerate(row):
                if tile_pos is None:
                    tile_pos = default_tile_idx
                if (tile_pos[0] + 1) * self.tile_size[0] > template.shape[-1]:
                    continue
                if (tile_pos[1] + 1) * self.tile_size[1] > template.shape[-2]:
                    continue

                template_patch = template[
                    :,
                    tile_pos[1] * self.tile_size[1]: (tile_pos[1] + 1) * self.tile_size[1],
                    tile_pos[0] * self.tile_size[0]: (tile_pos[0] + 1) * self.tile_size[0],
                ]

                if overlap == (0, 0):
                    image[
                        :,
                        y * self.tile_size[1]: (y + 1) * self.tile_size[1],
                        x * self.tile_size[0]: (x + 1) * self.tile_size[0],
                    ] = template_patch
                else:
                    sy = slice(y * (self.tile_size[1] - overlap[-2]), (y + 1) * (self.tile_size[1] - overlap[-2]) + overlap[-2])
                    sx = slice(x * (self.tile_size[0] - overlap[-1]), (x + 1) * (self.tile_size[0] - overlap[-1]) + overlap[-1])
                    image[:, sy, sx] = image[:, sy, sx] + window * template_patch
                    accum[:, sy, sx] = accum[:, sy, sx] + window

        if overlap != (0, 0):
            mask = accum > 0
            image[mask] = image[mask] / accum[mask]

        return image
