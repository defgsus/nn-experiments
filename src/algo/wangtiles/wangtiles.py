import math
from typing import List, Set, Iterable, Tuple, Optional

import torch
from torchvision.utils import make_grid

from .wangtemplate import WangTemplate
from .render import render_wang_tile


class WangTiles:

    Top = 0
    TopRight = 1
    Right = 2
    BottomRight = 3
    Bottom = 4
    BottomLeft = 5
    Left = 6
    TopLeft = 7

    map_offsets = [
        [-1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1],
        [ 1,  0],
        [ 1, -1],
        [ 0, -1],
        [-1, -1],
    ]

    class Tile:

        def __init__(self, parent: "WangTiles", index: int, colors: List[int]):
            self.parent = parent
            self.index = index
            self.colors = colors
            self._matching_indices = {}

        def __repr__(self):
            return f"Tile({self.index}, {self.colors})"

        @property
        def top(self): return self.colors[WangTiles.Top]

        @property
        def top_right(self): return self.colors[WangTiles.TopRight]

        @property
        def right(self): return self.colors[WangTiles.Right]

        @property
        def bottom_right(self): return self.colors[WangTiles.BottomRight]

        @property
        def bottom(self): return self.colors[WangTiles.Bottom]

        @property
        def bottom_left(self): return self.colors[WangTiles.BottomLeft]

        @property
        def left(self): return self.colors[WangTiles.Left]

        @property
        def top_left(self): return self.colors[WangTiles.TopLeft]

        @property
        def matching_indices_top(self) -> Set[int]:
            return self.matching_indices(WangTiles.Top)

        @property
        def matching_indices_top_right(self) -> Set[int]:
            return self.matching_indices(WangTiles.TopRight)

        @property
        def matching_indices_right(self) -> Set[int]:
            return self.matching_indices(WangTiles.Right)

        @property
        def matching_indices_bottom_right(self) -> Set[int]:
            return self.matching_indices(WangTiles.BottomRight)

        @property
        def matching_indices_bottom(self) -> Set[int]:
            return self.matching_indices(WangTiles.Bottom)

        @property
        def matching_indices_bottom_left(self) -> Set[int]:
            return self.matching_indices(WangTiles.BottomLeft)

        @property
        def matching_indices_left(self) -> Set[int]:
            return self.matching_indices(WangTiles.Left)

        @property
        def matching_indices_top_left(self) -> Set[int]:
            return self.matching_indices(WangTiles.TopLeft)

        def matches_top(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.Top)

        def matches_top_right(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.TopRight)

        def matches_right(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.Right)

        def matches_bottom_right(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.BottomRight)

        def matches_bottom(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.Bottom)

        def matches_bottom_left(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.BottomLeft)

        def matches_left(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.Left)

        def matches_top_left(self, tile: "WangTiles.Tile") -> bool:
            return self.matches(tile, WangTiles.TopLeft)

        def matches(self, tile: "WangTiles.Tile", direction: int) -> bool:
            return self.colors[direction] >= 0 and self.colors[direction] == tile.colors[(direction + 4) % 8]

        def matching_indices(self, direction: int) -> Set[int]:
            return self._matching_indices[direction]

    def __init__(
            self,
            colors: Iterable[Iterable[int]],
            mode: str = "edge",
    ):
        if mode in ("e", "edge"):
            self.mode = "edge"
        elif mode in ("c", "corner"):
            self.mode = "corner"
        elif mode in ("ec", "ce", "edgecorner", "corneredge"):
            self.mode = "edgecorner"
        else:
            raise ValueError(f"`mode` must be one of e, edge, c, corner, ec, ce, edgecorner or corneredge, got '{mode}'")

        self.tiles = []

        expected_length = 8 if self.mode == "edgecorner" else 4
        for idx, row in enumerate(colors):
            row = list(row)
            if len(row) != expected_length:
                raise ValueError(f"Item #{idx} in `colors` has length {len(row)}, expected {expected_length}")

            if self.mode == "edgecorner":
                colors = row

            elif self.mode == "edge":
                colors = [-1] * 8
                for i, c in enumerate(row):
                    colors[i * 2] = c

            else:  # "corner":
                colors = [-1] * 8
                for i, c in enumerate(row):
                    colors[i * 2 + 1] = c

            self.tiles.append(self.Tile(
                parent=self,
                index=idx,
                colors=colors,
            ))

        for tile1 in self.tiles:
            for direction in range(8):
                tile1._matching_indices[direction] = set()
                for tile2 in self.tiles:
                    if tile1.matches(tile2, direction):
                        tile1._matching_indices[direction].add(tile2.index)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Tile:
        return self.tiles[idx]

    def create_template(
            self,
            tile_shape: Tuple[int, int],
            padding: float = 0.,
            fade: float = 1.0,
            image: Optional[torch.Tensor] = None,
    ):
        nrow = int(math.sqrt(len(self.tiles)))

        if image is None:
            tile_images = []
            for i, tile in enumerate(self.tiles):
                tile_images.append(
                    render_wang_tile(
                        assignments=tile.colors,
                        shape=tile_shape,
                        padding=padding,
                        fade=fade,
                    )
                )

            image = make_grid(
                tile_images,
                padding=0,
                nrow=nrow,
            )

        indices = torch.linspace(0, nrow * nrow - 1, nrow * nrow).to(torch.int64).view(nrow, nrow)
        indices[indices >= len(self.tiles)] = -1
        return WangTemplate(indices, image)
