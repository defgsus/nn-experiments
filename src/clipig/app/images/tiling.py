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

        @property
        def edge_colors(self) -> Tuple[int, ...]:
            return self.t, self.l, self.b, self.r

        @property
        def corner_colors(self) -> Tuple[int, ...]:
            return self.tl, self.bl, self.br, self.tr

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
        attributes_map = {
            tuple(int(i) for i in key.split(",")): self.Attributes(**value)
            for key, value in config["attributes_map"].items()

        }
        self._attributes_map = {
            key: value for key, value in attributes_map.items()
            if all(k >= 0 for k in key)  # fix illegal value
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
    def map_size(self) -> Tuple[int, int]:
        if not self._attributes_map:
            return 0, 0
        return (
            max(x for x, y in self._attributes_map.keys()) + 1,
            max(y for x, y in self._attributes_map.keys()) + 1,
        )

    @property
    def attributes_map(self) -> Dict[Tuple[int, int], "LImageTiling.Attributes"]:
        return self._attributes_map

    def set_attributes_map(self, attributes_map: Dict[Tuple[int, int], "LImageTiling.Attributes"]):
        self._attributes_map = deepcopy(attributes_map)

    def set_optimal_attributes_map(self, num_colors: int = 2, mode: str = "edge"):
        from src.algo import wangtiles
        from src.algo.wangtiles2 import WangTiles2

        if num_colors == 1 and mode == "edge+corner":
            self._attributes_map.clear()
            for y, row in enumerate(WangTiles2.LAYOUTS["edge-corner-7x7"]):
                for x, v in enumerate(row):
                    attr_kwargs = {}
                    for key in ("TL", "T", "TR", "L", "R", "BL", "B", "BR"):
                        attr_kwargs[key.lower()] = int(bool(v & getattr(WangTiles2, key))) - 1
                    self._attributes_map[(x, y)] = self.Attributes(**attr_kwargs)
            return

        opt_indices = wangtiles.OPTIMAL_WANG_INDICES_SQUARE.get((mode, num_colors))
        if opt_indices is None:
            raise NotImplementedError(f"Sorry, no optimal tile indices for {num_colors}-{mode}")
        wang_tiles = wangtiles.WangTiles(colors=wangtiles.get_wang_tile_colors(num_colors), mode=mode)

        self._attributes_map.clear()
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

        self.resolve_map(map, seed=seed)
        return map

    def create_map_stochastic_perlin_islands(
            self,
            size: Tuple[int, int],
            island_size: float = .5,
            island_cells: int = 2,
            random_prob: float = .1,
            seed: Optional[int] = None,
    ):
        from src.algo.sdf.two_d.util import perlin_noise_2d

        island_size = 1 - island_size * 2
        rng = random.Random(seed)

        elements_tiles = [
            (a.num_elements(), idx)
            for idx, a in self.attributes_map.items()
        ]
        elements_tiles.sort(key=lambda t: t[0])
        tiles_outside = [t[1] for t in elements_tiles if t[0] == elements_tiles[0][0]]
        tiles_inside = [t[1] for t in elements_tiles if t[0] == elements_tiles[-1][0]]

        noise = perlin_noise_2d(
            shape=(size[1], size[0]),
            res=(
                max(1, int(size[1] / island_cells)),
                max(1, int(size[0] / island_cells)),
            ),
            rng=np.random.RandomState(seed),
        )

        map = np.tile(np.array([[None]], dtype=np.object_), noise.shape)
        for y, row in enumerate(map):
            for x, v in enumerate(row):
                if noise[y, x] > island_size:
                    map[y, x] = repr(rng.choice(tiles_inside))

        all_tiles = list(self.attributes_map.keys())
        for i in range(int(size[0]*size[1] * random_prob)):
            x, y = rng.randrange(size[0]), rng.randrange(size[1])
            if map[y, x] is None or rng.uniform(0, 1) < random_prob:
                map[y, x] = repr(rng.choice(all_tiles))

        map_list = []
        for row in map:
            row_list = []
            for x in row:
                if x is None:
                    x = rng.choice(tiles_outside)
                else:
                    x = eval(x)
                row_list.append(x)
            map_list.append(row_list)
        self.resolve_map(map_list)
        return map_list

    def set_map_default(self, map: List[List[Optional[Tuple[int, int]]]], default: Tuple[int, int] = (0, 0)):
        for row in map:
            for x, v in enumerate(row):
                if v is None:
                    row[x] = default

    def tile_num_map_mismatches(
            self,
            map: List[List[Tuple[int, int]]],
            attrs: Attributes,
            x: int,
            y: int,
            min_color: int = 0,
    ) -> int:
        count = 0
        if x > 0:
            other = self.tile_attributes(map[y][x-1])
            if attrs.l >= min_color and attrs.l != other.r:
                count += 1
            if attrs.tl >= min_color and attrs.tl != other.tr:
                count += 1
            if attrs.bl >= min_color and attrs.bl != other.br:
                count += 1
        if x < len(map[0]) - 1:
            other = self.tile_attributes(map[y][x+1])
            if attrs.r >= min_color and attrs.r != other.l:
                count += 1
            if attrs.tr >= min_color and attrs.tr != other.tl:
                count += 1
            if attrs.br >= min_color and attrs.br != other.bl:
                count += 1
        if y > 0:
            other = self.tile_attributes(map[y-1][x])
            if attrs.t >= min_color and attrs.t != other.b:
                count += 1
            if attrs.tl >= min_color and attrs.tl != other.bl:
                count += 1
            if attrs.tr >= min_color and attrs.tr != other.br:
                count += 1
        if y < len(map) - 1:
            other = self.tile_attributes(map[y+1][x])
            if attrs.b >= min_color and attrs.b != other.t:
                count += 1
            if attrs.bl >= min_color and attrs.bl != other.tl:
                count += 1
            if attrs.br >= min_color and attrs.br != other.tr:
                count += 1

        return count

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
            num_iterations: int = 23,
            seed: Optional[int] = None,
    ) -> bool:
        rng = random.Random(seed)
        def _iter_mismatches():
            positions = [(x, y) for y in range(len(map)) for x in range(len(map[0]))]
            rng.shuffle(positions)
            for x, y in positions:
                if map[y][x] is not None:
                    attrs = self.attributes_map[map[y][x]]
                    if (count := self.tile_num_map_mismatches(map, attrs, x, y)) > 0:
                        yield x, y

        def _find_tile(x, y):
            tiles = []
            for tile_idx, attrs in self.attributes_map.items():
                count = self.tile_num_map_mismatches(map, attrs, x, y)
                tiles.append((count, tile_idx))

            rng.shuffle(tiles)
            tiles.sort(key=lambda t: t[0])
            return tiles[0][1]

        for it in range(max(1, num_iterations)):
            has_errors = False
            for x, y in _iter_mismatches():
                has_errors = True
                t = _find_tile(x, y)
                if t is not None:
                    map[y][x] = t

            if not has_errors:
                return True

        return not has_errors

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
            (template.shape[-3],
             len(map) * self.tile_size[1],
             len(map[0]) * self.tile_size[0]
            ),
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
