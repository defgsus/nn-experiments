import math

import torch
import torch.nn as nn


# https://github.com/icey-zhang/simple_diffusion_example/blob/main/diffusion.py
class SinusoidalNumberEmbedding(nn.Module):
    def __init__(self, dim: int, period: int = 100):
        super().__init__()
        assert dim >= 4
        assert dim % 2 == 0
        self.dim = dim
        self.period = period

    def forward(self, number: torch.Tensor):
        if number.ndim == 1:
            number = number.unsqueeze(-1)
        elif number.ndim > 2:
            raise ValueError(f"Expect number.ndim to be 1 or 2, got {number.shape}")

        device = number.device
        half_dim = self.dim // 2
        embeddings = math.log(self.period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = number * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
