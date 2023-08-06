from typing import Tuple

import torch
import torch.nn as nn


class FreescaleImageModule(nn.Module):

    def __init__(self, num_in: int):
        super().__init__()
        self.num_in = num_in

    def forward_state(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        assert x.ndim == 2, f"Got {x.shape}"
        assert len(shape) == 3, f"Got {shape}"

        batch_size = x.shape[0]

        coords = (
            torch.linspace(-1, 1, shape[-1], dtype=x.dtype, device=x.device)
                .view(-1, 1)
                .repeat(shape[-2], 2)
        )
        c2 = (
            torch.linspace(-1, 1, shape[-2], dtype=x.dtype, device=x.device)
                .view(-1, 1)
                .repeat(1, shape[-1])
                .view(shape[-2] * shape[-1], 1)
        )
        coords[:, 1:2] = c2

        state = torch.cat(
            [
                x.repeat(shape[-2] * shape[-1], 1),
                coords.repeat(batch_size, 1),
            ],
            dim=1,
        )

        state = self.forward_state(state, shape)

        y = state.view(batch_size, shape[-2], shape[-1], shape[-3]).permute(0, 3, 2, 1)

        y = torch.clip(y, 0, 1)

        return y
