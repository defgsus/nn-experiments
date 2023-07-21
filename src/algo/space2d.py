import torch
from typing import Optional, Callable, List, Tuple, Iterable, Generator


class Space2d:

    def __init__(
            self,
            shape: Tuple[int, int, int],
            offset: Optional[torch.Tensor] = None,
            scale: float = 1.,
            dtype: torch.dtype = torch.float32,
    ):
        """
        shape: [DIMS, HEIGHT, WIDTH]
        """
        self.shape = shape
        self.offset = offset
        self.scale = scale
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
        space *= self.scale
        if self.offset is not None:
            space += self.offset.reshape(-1, 1, 1)
        return space
