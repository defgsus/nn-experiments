import math
from typing import Optional, Callable, List, Tuple, Iterable, Generator

import torch


class Space2d:

    def __init__(
            self,
            shape: Tuple[int, int, int],
            offset: Optional[torch.Tensor] = None,
            scale: float = 1.,
            rotate_2d: Optional[float] = None,
            dtype: torch.dtype = torch.float32,
    ):
        """
        shape: [DIMS, HEIGHT, WIDTH]
        """
        if rotate_2d is not None:
            assert shape[0] == 2, f"Expected 2 dims for rotate_2d, got {shape[0]}"

        self.shape = shape
        self.offset = offset
        self.scale = scale
        self.rotate_2d = rotate_2d
        self.dtype = dtype

    def space(self) -> torch.Tensor:
        space = torch.zeros(self.shape)
        space[0, :, :] = (
            torch.linspace(-1, 1, self.shape[-1])
                .reshape(1, -1)
                .repeat((self.shape[-2], 1))
        )
        space[1, :, :] = (
            torch.linspace(-1, 1, self.shape[-2])
                .reshape(1, -1)
                .repeat((self.shape[-1], 1))
                .permute(1, 0)
        )

        if self.rotate_2d is not None:
            si = math.sin(self.rotate_2d)
            co = math.cos(self.rotate_2d)
            space = torch.cat([
                (co * space[1] + si * space[0]).unsqueeze(0),
                (co * space[0] - si * space[1]).unsqueeze(0)
            ])

        space *= self.scale

        if self.offset is not None:
            space += self.offset.reshape(-1, 1, 1)

        return space
