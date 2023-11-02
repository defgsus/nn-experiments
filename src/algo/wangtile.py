import random
from typing import Tuple, Union

import torch

from src.util.image import get_image_window


# Top, Right, Bottom, Left
WANG_TILES = [
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
]

# the classic 4x4 template
WANG_TEMPLATE_INDICES = [
    4, 6, 14, 12,
    5, 7, 15, 13,
    1, 3, 11, 9,
    0, 2, 10, 8,
]

WANG_INDEX_TO_TEMPLATE_INDEX = {
    tile_idx: template_idx
    for template_idx, tile_idx in enumerate(WANG_TEMPLATE_INDICES)
}


def render_wang_map(
        wang_template: torch.Tensor,
        tile_indices: torch.Tensor,
        overlap: Union[int, Tuple[int, int]] = 0,
):
    if isinstance(overlap, int):
        overlap = [overlap, overlap]

    for s in wang_template.shape[-2:]:
        if not s % 4 == 0:
            raise ValueError(f"`wang_template` size must be divisible by 4, got {wang_template.shape}")

    tile_size_y = wang_template.shape[-2] // 4
    tile_size_x = wang_template.shape[-1] // 4

    if overlap[-2] > tile_size_y or overlap[-1] > tile_size_x:
        raise ValueError(
            f"`overlap` exceeds tile size, got {overlap}, tile size is {[tile_size_y, tile_size_x]}"
        )

    image = torch.zeros(
        wang_template.shape[-3],
        tile_indices.shape[-2] * (tile_size_y - overlap[-2]) + overlap[-2],
        tile_indices.shape[-1] * (tile_size_x - overlap[-1]) + overlap[-1],
    ).to(wang_template)

    if overlap != (0, 0):
        accum = torch.zeros_like(image)
        window = get_image_window((tile_size_y, tile_size_x)).to(image)
        
    for y, row in enumerate(tile_indices):
        for x, tile_idx in enumerate(row):
            t_idx = WANG_INDEX_TO_TEMPLATE_INDEX[int(tile_idx)]
            tx = t_idx % 4
            ty = t_idx // 4
            template_patch = wang_template[
                             :,
                             ty * tile_size_y: (ty + 1) * tile_size_y,
                             tx * tile_size_x: (tx + 1) * tile_size_x,
                             ]
            if overlap == (0, 0):
                image[
                :,
                y * tile_size_y: (y + 1) * tile_size_y,
                x * tile_size_x: (x + 1) * tile_size_x,
                ] = template_patch
            else:
                sy = slice(y * (tile_size_y - overlap[-2]), (y + 1) * (tile_size_y - overlap[-2]) + overlap[-2])
                sx = slice(x * (tile_size_x - overlap[-1]), (x + 1) * (tile_size_x - overlap[-1]) + overlap[-1])
                image[:, sy, sx] = image[:, sy, sx] + window * template_patch
                accum[:, sy, sx] = accum[:, sy, sx] + window

    if overlap != (0, 0):
        mask = accum > 0
        image[mask] = image[mask] / accum[mask]

    return image


def random_wang_map(shape: Tuple[int, int]) -> torch.Tensor:
    tiles = [
        [None] * shape[-1]
        for _ in range(shape[-2])
    ]
    for y in range(shape[-2]):
        for x in range(shape[-1]):

            while True:
                tile_idx = random.randrange(len(WANG_TILES))
                if x >= 1 and tiles[y][x - 1] is not None:
                    if WANG_TILES[tiles[y][x - 1]][1] != WANG_TILES[tile_idx][3]:
                        continue
                if y >= 1 and tiles[y - 1][x] is not None:
                    if WANG_TILES[tiles[y - 1][x]][2] != WANG_TILES[tile_idx][0]:
                        continue

                break

            tiles[y][x] = tile_idx

    return torch.Tensor(tiles).to(torch.int64)
