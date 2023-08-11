import math
from typing import Optional, Union, Iterable

import torch

from .space2d import Space2d


def menger_sponge_2d(
        space: Space2d,
        iterations: int = 3,
        radius: float = .25,
        repeat_size: Union[float, Iterable[float]] = 1.,
        rotate_z_deg: float = 45.,
        scale_factor: float = 2.,
        offset: Iterable[float] = (0., 0.),
        shape: str = "square",  # "square", "circle", "torus", "stripe"
        aa: int = 1,
) -> torch.Tensor:
    offset = torch.Tensor(list(offset)).reshape(2, 1, 1).to(space.dtype)
    if not isinstance(repeat_size, (int, float)):
        repeat_size = torch.Tensor(list(repeat_size)).reshape(2, 1, 1).to(space.dtype)
    rotate_z = rotate_z_deg * 3.14159265 / 180.

    def _render(coords: torch.Tensor):
        dist_accum = torch.empty(1, *coords.shape[-2:], dtype=space.dtype).fill_(100000.)

        for iteration in range(iterations):
            l_coords = (coords + repeat_size * .5) % repeat_size - repeat_size * .5

            if shape in ("circle", "torus"):
                dist = torch.sqrt(torch.sum(torch.square(l_coords), dim=0, keepdim=True))
                dist1, _ = torch.min(dist - radius, dim=0, keepdim=True)
                dist = dist1
                if shape == "torus":
                    dist2, _ = torch.min(dist - radius * .5, dim=0, keepdim=True)
                    dist = torch.maximum(-dist1, dist2)

            elif shape in ("square", "stripe"):
                if shape == "stripe":
                    dist = torch.abs(l_coords[0]).unsqueeze(0)
                else:
                    dist = torch.abs(l_coords)
                dist, _ = torch.max(dist - radius, dim=0, keepdim=True)

            else:
                raise ValueError(f"Unknown shape '{shape}")

            if iteration == 0:
                dist_accum = torch.minimum(dist_accum, -dist)
            else:
                dist_accum = torch.maximum(dist_accum, -dist)

            si = math.sin(rotate_z)
            co = math.cos(rotate_z)
            coords = torch.cat([
                (co * coords[1] + si * coords[0]).unsqueeze(0),
                (co * coords[0] - si * coords[1]).unsqueeze(0)
            ])
            coords = coords * scale_factor
            coords = coords + offset

        output = 1. - dist_accum * 100.
        return torch.clamp(output, 0, 1)

    if aa <= 1:
        return _render(space.space())
    else:
        return space.reduce_aa_output(aa, _render(space.aa_space(aa)))
