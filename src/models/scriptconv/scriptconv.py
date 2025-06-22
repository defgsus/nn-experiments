from typing import Tuple

import torch
import torch.nn as nn


class ScriptConvModel(nn.Module):

    def __init__(
            self,
            script: str,
            input_shape: Tuple[int, int, int],
    ):
        from .parser import create_layers
        super().__init__()
        self.layers = create_layers(script, input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
