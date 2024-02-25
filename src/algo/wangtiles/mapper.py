import random
from typing import Tuple, Optional, Iterable, Dict

import torch

from .wangtiles import WangTiles


def wang_map_stochastic_scanline(
        wang_tiles: WangTiles,
        shape: Tuple[int, int],
        include: Optional[Iterable[int]] = None,
        exclude: Optional[Iterable[int]] = None,
        probabilities: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    if include is not None:
        include = set(include)
    if exclude is not None:
        exclude = set(exclude)

    possible_tiles = include.copy() if include is not None else set(range(len(wang_tiles.tiles)))
    if exclude is not None:
        possible_tiles -= exclude

    possible_tiles = sorted(possible_tiles)
    tile_probs = {
        t: 1.
        for t in possible_tiles
    }
    if probabilities:
        for idx, prob in probabilities.items():
            if idx in possible_tiles:
                tile_probs[idx] = prob

    min_prob = None
    for prob in tile_probs.values():
        if prob > 0 and (min_prob is None or prob < min_prob):
            min_prob = prob

    if min_prob is None:
        raise ValueError(f"All tile probabilities are zero")

    possible_tiles_lookup = []
    for t in possible_tiles:
        count = max(1, int(tile_probs[t] / min_prob))
        possible_tiles_lookup.extend([t] * count)

    # print(possible_tiles_lookup)

    tiles = [
        [-1] * shape[-1]
        for _ in range(shape[-2])
    ]
    for y in range(shape[-2]):
        for x in range(shape[-1]):

            random.shuffle(possible_tiles_lookup)
            for tile_idx in possible_tiles_lookup:

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

    return torch.Tensor(tiles).to(torch.int64)
