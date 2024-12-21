from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_callable


def dpfp(x: torch.Tensor, nu: int = 1) -> torch.Tensor:
    """
    Deterministic Parameter-Free Projection (DPFP)

    From Appendix C, "Linear Transformers Are Secretly Fast Weight Programmers"

    https://arxiv.org/abs/2102.11174

    :param x: Tensor of shape [..., N]
    :param nu: int
    :return: Tensor of shape [..., N * 2 * nu]
    """
    x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
    x_rolled = torch.cat([
        x.roll(shifts=j, dims=-1)
        for j in range(1, nu+1)
    ], dim=-1)
    x_repeat = torch.cat([x] * nu, dim=-1)
    return x_repeat * x_rolled


class Attention1d(nn.Module):
    def __init__(
            self,
            activation: Union[None, str, Callable] = "elu+1",
    ):
        super().__init__()
        self._activation_param = activation
        if activation == "elu+1":
            self.activation = lambda x: F.elu(x) + 1.
        elif activation == "dpfp":
            self.activation = dpfp
        else:
            self.activation = activation_to_callable(activation)

    def extra_repr(self) -> str:
        return f"activation={repr(self._activation_param)}"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.activation:
            q = self.activation(q)
            k = self.activation(k)
        return q @ (k.permute(0, 2, 1) @ v) / (v.shape[-1] * v.shape[-2])



class LinearAttention2d(nn.Module):
    """
    from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, Q, K, V):
        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, V)

        # Compute the normalizer
        Z = 1. / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()
