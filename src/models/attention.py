import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.
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
