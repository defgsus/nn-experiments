import random
from typing import Tuple, Optional, Iterable

import torch

from .wangtiles import WangTiles


def wang_map_stochastic_scanline(
        wang_tiles: WangTiles,
        shape: Tuple[int, int],
        include: Optional[Iterable[int]] = None,
        exclude: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    if include is not None:
        include = set(include)
    if exclude is not None:
        exclude = set(exclude)

    possible_tiles_set = include.copy() if include is not None else set(range(len(wang_tiles.tiles)))
    if exclude is not None:
        possible_tiles_set -= exclude

    tiles = [
        [-1] * shape[-1]
        for _ in range(shape[-2])
    ]
    for y in range(shape[-2]):
        for x in range(shape[-1]):

            possible_tiles = list(possible_tiles_set)
            random.shuffle(possible_tiles)
            for tile_idx in possible_tiles:

                if "edge" in wang_tiles.mode:
                    if x >= 1 and tiles[y][x - 1] >= 0:
                        if not wang_tiles[tile_idx].matches_left(wang_tiles[tiles[y][x - 1]]):
                            continue
                    if y >= 1 and tiles[y - 1][x] >= 0:
                        if not wang_tiles[tile_idx].matches_top(wang_tiles[tiles[y - 1][x]]):
                            continue

                if "corner" in wang_tiles.mode:
                    if x >= 1 and tiles[y][x - 1] >= 0:
                        if wang_tiles[tile_idx].colors[WangTiles.TopLeft] != wang_tiles[tiles[y][x - 1]].colors[WangTiles.TopRight]:
                            continue
                        if wang_tiles[tile_idx].colors[WangTiles.BottomLeft] != wang_tiles[tiles[y][x - 1]].colors[WangTiles.BottomRight]:
                            continue

                    if y >= 1 and tiles[y - 1][x] >= 0:
                        if wang_tiles[tile_idx].colors[WangTiles.TopLeft] != wang_tiles[tiles[y - 1][x]].colors[WangTiles.BottomLeft]:
                            continue
                        if wang_tiles[tile_idx].colors[WangTiles.TopRight] != wang_tiles[tiles[y - 1][x]].colors[WangTiles.BottomRight]:
                            continue

                    if x >= 1 and y >= 1 and tiles[y - 1][x - 1] >= 0:
                        if not wang_tiles[tile_idx].matches_top_left(wang_tiles[tiles[y - 1][x - 1]]):
                            continue

                tiles[y][x] = tile_idx or 0
                break

    return torch.Tensor(tiles).to(torch.int64)#.clamp_min(0)
