from typing import Optional

import torch

from .space2d import Space2d


def kali2d(
        space: Space2d,
        param: torch.Tensor,
        iterations: int = 7,
        out_weights: Optional[torch.Tensor] = None,
        accumulate: str = "mean",  # mean, max, min, none
        aa: int = 0,
) -> torch.Tensor:
    param = param.reshape(-1, 1, 1)

    def _render(space: torch.Tensor) -> torch.Tensor:
        if accumulate == "none":
            pass
        if accumulate == "min":
            accum = torch.ones_like(space) * iterations
        else:
            accum = torch.zeros_like(space)

        for iteration in range(iterations):
            #dot_prod = space.sum(dim=0).unsqueeze(0).repeat(3, 1, 1)
            dot_prod = torch.sum(space * space, dim=0, keepdim=True) + 0.000001
            space = torch.abs(space) / dot_prod

            if accumulate == "mean":
                accum = accum + space

            elif accumulate == "max":
                accum = torch.max(space, accum)

            elif accumulate == "min":
                accum = torch.min(space, accum)

            if iteration < iterations - 1:
                space = space - param

        if accumulate == "none":
            output = space # iterations
        elif accumulate == "min":
            output = accum * iterations
        else:
            output = accum / iterations

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
        #output = output[:, (aa-1)*s[-2]:(aa)*s[-2], (aa-1)*s[-1]:(aa)*s[-1]]

    else:
        output = _render(space.space())

    if out_weights is not None:
        a = output.permute(1, 2, 0).reshape(-1, 3)
        output = torch.matmul(a, out_weights).reshape((output.shape[1], output.shape[2], output.shape[0])).permute(2, 0, 1)

    return torch.clamp(output, 0, 1)
