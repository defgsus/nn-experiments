"""
Nice kali params:

- 2-dim, plane-x
    - 1.13, 0.43
    - 0.92, 1.13
    - 1., 0.16
    - 0.48, 0.16
    - 0.476, 0.1195
    - 0.41, 0.09
    - 0.63, 0.25
    - 0.6, 0.05
    - 0.71, 0.16
    - 0.61, 1.62
    - 0.52, 1.555
    - 0.05, 1.53
    - 0.815, 1.61
    - 0.31, 1.06
    - 0.27, 1.04
    - 0.22, 1.026
    - 0.18, 1.01
"""
from typing import Union, Iterable, Optional, Literal, List

import torch
import torch.nn as nn

from .fractal_base import FractalBaseLayer

class KaliSetLayer(FractalBaseLayer):

    ACCUMULATION_TYPES = ["none", "mean", "max", "min", "submin", "alternate"]

    def __init__(
            self,
            param: Union[torch.Tensor, Iterable[float]],
            iterations: int = 1,
            axis: int = -1,
            scale: Union[None, float, Iterable[float], torch.Tensor] = None,
            offset: Union[None, Iterable[float], torch.Tensor] = None,
            accumulate: Literal["none", "mean", "max", "min", "submin", "alternate"] = "none",
            exponent: Optional[float] = None,
            mixer: Union[None, torch.Tensor, List[List[float]]] = None,
            learn_param: bool = False,
            learn_mixer: bool = False,
            learn_scale: bool = False,
            learn_offset: bool = False,
    ):
        """
        A layer that calculates the kali-set.

        See:
            https://www.fractalforums.com/new-theories-and-research/very-simple-formula-for-fractal-patterns
            https://defgsus.github.io/blog/2021/10/12/kaliset-explorer.html

        Think of the input as coordinates.

        The module accepts any shape as long as the `axis` dimension has the same size as `param`.
        And `param` should have at least shape (2,)

        :param param: magic parameter
        :param iterations: number of iterations
        :param axis: int, axis of the channels
        :param learn_param: do train the parameter
        :param mixer: matrix of shape (len(param), len(param)) to transform the final values
        """
        if not isinstance(param, torch.Tensor):
            param = torch.Tensor(param)
        if param.ndim != 1:
            raise ValueError(f"Expected `param` to have 1 dimension, got shape {param.shape}")

        super().__init__(
            num_channels=param.shape[0],
            axis=axis,
            scale=scale,
            offset=offset,
            mixer=mixer,
            learn_mixer=learn_mixer,
            learn_scale=learn_scale,
            learn_offset=learn_offset,
        )
        self.iterations = iterations
        self.exponent = exponent
        self.accumulate = accumulate

        self.param = nn.Parameter(param, requires_grad=learn_param)

    def fractal(self, x: torch.Tensor, axis: int) -> torch.Tensor:

        slices = [None] * (x.ndim - axis - 1)
        param = self.param[..., *slices]

        if self.accumulate == "none":
            pass
        elif self.accumulate in ("min", "submin"):
            accum = torch.ones_like(x) * self.iterations
        else:
            accum = torch.zeros_like(x)

        for i in range(self.iterations):
            dot_prod = torch.sum(x ** 2, dim=axis, keepdim=True) + 0.000001
            x = x.abs() / dot_prod

            a_x = x
            if self.exponent is not None:
                a_x = torch.exp(-a_x * self.exponent)

            if self.accumulate == "mean":
                accum = accum + a_x

            elif self.accumulate == "max":
                accum = torch.max(a_x, accum)

            elif self.accumulate == "min":
                accum = torch.min(a_x, accum)

            elif self.accumulate == "submin":
                accum = accum - torch.min(accum, a_x)

            elif self.accumulate == "alternate":
                accum = accum + (a_x if self.iterations % 2 == 0 else -a_x)

            if i + 1 < self.iterations:
                x = x - param

        if self.accumulate == "none":
            output = a_x
        elif self.accumulate == "min":
            output = accum * self.iterations
        else:
            output = accum / self.iterations
            if self.accumulate == "alternate":
                output = output * 2

        return output
