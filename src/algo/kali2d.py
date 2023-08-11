from typing import Optional

import torch

from .space2d import Space2d


def kali_2d(
        space: Space2d,
        param: torch.Tensor,
        iterations: int = 7,
        out_weights: Optional[torch.Tensor] = None,
        accumulate: str = "none",  # none, mean, max, min, submin, alternate
        exponent: float = 0,
        sin_freq: float = 0,
        aa: int = 0,
) -> torch.Tensor:
    param = param.reshape(-1, 1, 1)

    def _render(space: torch.Tensor) -> torch.Tensor:
        if accumulate == "none":
            pass
        elif accumulate in ("min", "submin"):
            accum = torch.ones_like(space) * iterations
        else:
            accum = torch.zeros_like(space)

        for iteration in range(iterations):
            dot_prod = torch.sum(space * space, dim=0, keepdim=True) + 0.000001
            space = torch.abs(space) / dot_prod

            a_space = space
            if exponent:
                a_space = torch.exp(-a_space * exponent)

            if sin_freq:
                a_space = torch.sin(a_space * sin_freq)

            if accumulate == "mean":
                accum = accum + a_space

            elif accumulate == "max":
                accum = torch.max(a_space, accum)

            elif accumulate == "min":
                accum = torch.min(a_space, accum)

            elif accumulate == "submin":
                accum = accum - torch.min(accum, a_space)

            elif accumulate == "alternate":
                accum = accum + (a_space if iteration % 2 == 0 else -a_space)

            if iteration < iterations - 1:
                space = space - param

        if accumulate == "none":
            output = a_space
        elif accumulate == "min":
            output = accum * iterations
        else:
            output = accum / iterations
            if accumulate == "alternate":
                output = output * 2

        return output

    if aa and aa > 1:
        s = space.shape
        aa_space = space.space().repeat(1, aa, aa)
        for x in range(aa):
            for y in range(aa):
                if x or y:
                    aa_space[1, y*s[-2]:(y+1)*s[-2], x*s[-1]:(x+1)*s[-1]] += (y / aa / s[-2]) * space.scale
                    aa_space[0, y*s[-2]:(y+1)*s[-2], x*s[-1]:(x+1)*s[-1]] += (x / aa / s[-1]) * space.scale

        output = _render(aa_space)
        for x in range(0, aa):
            for y in range(0, aa):
                if x or y:
                    output[:, :s[-2], :s[-1]] = output[:, :s[-2], :s[-1]] + output[:, y*s[-2]:(y+1)*s[-2], x*s[-1]:(x+1)*s[-1]]
        output = output[:, :s[-2], :s[-1]] / (aa * aa)

    else:
        output = _render(space.space())

    if out_weights is not None:
        a = output.permute(1, 2, 0).reshape(-1, 3)
        output = torch.matmul(a, out_weights).reshape((output.shape[1], output.shape[2], output.shape[0])).permute(2, 0, 1)

    return torch.clamp(output, 0, 1)
