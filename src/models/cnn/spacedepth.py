"""
inspired by https://github.com/LabSAINT/SPD-Conv / https://arxiv.org/pdf/2208.03641.pdf
"""
import torch


def space_to_depth(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ],
        dim=1
    )


def depth_to_space(x: torch.Tensor) -> torch.Tensor:
    nc = x.shape[-3] // 2
    s = torch.stack([
        x[..., :nc, :, :],
        x[..., nc:, :, :],
        # d[..., 1:nc+1, :, :]
    ], dim=-1).view(-1, nc, x.shape[-2], x.shape[-1] * 2)

    nc //= 2
    return torch.stack([
        s[..., :nc, :, :],
        s[..., nc:, :, :],
    ], dim=-2).view(-1, nc, x.shape[-2] * 2, x.shape[-1] * 2)
