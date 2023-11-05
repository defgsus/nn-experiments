from typing import Tuple, Optional, Union

import torch

from src.algo.space2d import Space2d
from src.util.image import get_image_window


def smoothstep(a, b, x):
    x = ((x - a) / (b - a)).clamp(0, 1)
    return x * x * (3. - 2. * x)


def render_wang_tile(
        assignments: Tuple[int, int, int, int, int, int, int, int],
        shape: Tuple[int, int],
        padding: float = 0.,
        fade: float = 1.,
        colors: Optional[Tuple[Tuple[int, int, int], ...]] = None,
) -> torch.Tensor:
    if colors is None:
        colors = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    colors = torch.Tensor(colors)

    image = torch.zeros(3, *shape[-2:])
    space = Space2d((2, *shape[-2:])).space()
    space_r = Space2d((2, *shape[-2:]), rotate_2d=torch.pi / 4.).space()

    for idx, coords in enumerate([
        space[1] + 1.,
        1. - space_r[1],
        1. - space[0],
        1. - space_r[0],
        1. - space[1],
        space_r[1] + 1.,
        space[0] + 1.,
        space_r[0] + 1.,
    ]):
        if assignments[idx] >= 0:
            c = (coords - padding)
            c = smoothstep(fade, 0., c)
            image = torch.max(image, c.expand(3, -1, -1) * colors[assignments[idx]].reshape(3, 1, 1))

    return image.clamp(0, 1)


def render_wang_map(
        wang_template: "WangTemplate",
        tile_indices: torch.Tensor,
        overlap: Union[int, Tuple[int, int]] = 0,
):
    if isinstance(overlap, int):
        overlap = [overlap, overlap]

    tile_shape = wang_template.tile_shape

    if overlap[-2] > tile_shape[-2] or overlap[-1] > tile_shape[-1]:
        raise ValueError(
            f"`overlap` exceeds tile size, got {overlap}, tile shape is {tile_shape}"
        )

    image = torch.zeros(
        wang_template.shape[-3],
        tile_indices.shape[-2] * (tile_shape[-2] - overlap[-2]) + overlap[-2],
        tile_indices.shape[-1] * (tile_shape[-1] - overlap[-1]) + overlap[-1],
    )
    if overlap != (0, 0):
        accum = torch.zeros_like(image)
        window = get_image_window(tile_shape[-2:]).to(image)

    for y, row in enumerate(tile_indices):
        for x, tile_idx in enumerate(row):
            if tile_idx < 0:
                continue

            template_patch = wang_template.tile(tile_idx).to(image)

            if overlap == (0, 0):
                image[
                    :,
                    y * tile_shape[-2]: (y + 1) * tile_shape[-2],
                    x * tile_shape[-1]: (x + 1) * tile_shape[-1],
                ] = template_patch
            else:
                sy = slice(y * (tile_shape[-2] - overlap[-2]), (y + 1) * (tile_shape[-2] - overlap[-2]) + overlap[-2])
                sx = slice(x * (tile_shape[-1] - overlap[-1]), (x + 1) * (tile_shape[-1] - overlap[-1]) + overlap[-1])
                image[:, sy, sx] = image[:, sy, sx] + window * template_patch
                accum[:, sy, sx] = accum[:, sy, sx] + window

    if overlap != (0, 0):
        mask = accum > 0
        image[mask] = image[mask] / accum[mask]

    return image
