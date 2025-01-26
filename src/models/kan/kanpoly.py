from typing import Union, Callable

import torch
import torch.nn as nn

from src.models.util import activation_to_module


class KANPolyLayer(nn.Module):
    """
    based on https://github.com/SciYu/KAE/blob/main/DenseLayerPack/KAE.py
    """
    def __init__(
            self,
            input_dim: int,
            out_dim: int,
            order: int,
            bias: bool = True,
            activation: Union[None, str, Callable] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order + 1) * 0.01)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))
        self.act = activation_to_module(activation)

    def extra_repr(self):
        return f"input_dim={self.input_dim}, out_dim={self.out_dim}, order={self.order}, bias={self.bias is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)

        for i in range(self.order + 1):
            term = (x_expanded**i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.bias is not None:
            y = y + self.bias

        if self.act is not None:
            y = self.act(y)

        return y
