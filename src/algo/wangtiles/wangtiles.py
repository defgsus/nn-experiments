from typing import List, Set, Iterable


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

        self.mode = mode
        self.tiles = []

        expected_length = 8 if mode == "edgecorner" else 4
        for idx, row in enumerate(colors):
            row = list(row)
            if len(row) != expected_length:
                raise ValueError(f"Item #{idx} in `colors` has length {len(row)}, expected {expected_length}")
            if mode == "edgecorner":
                colors = row
            elif mode == "edge":
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

    def __getitem__(self, idx: int) -> Tile:
        return self.tiles[idx]


class WangTiles2E(WangTiles):
    """All wang tiles with 2 colors (0 or 1) on edges"""
    def __init__(self):
        super().__init__(
            colors=[
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            mode="edge",
        )


class WangTiles2C(WangTiles):
    """All wang tiles with 2 colors (0 or 1) on corners"""
    def __init__(self):
        super().__init__(
            colors=[
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            mode="corner",
        )
