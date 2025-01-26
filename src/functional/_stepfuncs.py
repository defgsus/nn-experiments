import math
from typing import Union, Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinStep(nn.Module):
    """
    Activation function that does `x + sin(x)` with some parameters

    Creates smooth steps along x.

    Not as good as ReLU, but close.
    """
    def __init__(
            self,
            factor: float = 1.,
            pre_activation: Union[None, str, Callable, Type[nn.Module]] = None,
    ):
        from src.models.util import activation_to_callable

        super().__init__()
        assert factor > 0, f"Got {self._factor}"
        self._factor = factor
        self._factor_pi = factor * math.pi * 2
        self._offset = .5 / factor
        self._pre_act = pre_activation
        self.pre_act = activation_to_callable(pre_activation)

    def extra_repr(self):
        return f"factor={self._factor}, pre_activation={repr(self._pre_act)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_act is not None:
            x = self.pre_act(x)
        return x + torch.sin((x + self._offset) * self._factor_pi) / self._factor_pi

    @classmethod
    def create_from_string(cls, name: str) -> "SinStep":
        """
        Create new instance

        :param name: str, can be
            - "sinstep"
            - "sinstep<factor>"
            - "sinstep<factor><pre_activation>"
        :return: SinStep instance
        """
        n = name.lower()
        if not n.startswith("sinstep"):
            raise ValueError(f"Invalid name for SinStep '{name}'")
        n = n[7:]

        factor = ""
        for ch in n:
            if ch.isdigit() or ch == ".":
                factor = factor + ch
            else:
                break

        return cls(
            factor=float(factor),
            pre_activation=n[len(factor):] or None,
        )


class SigmoidStep(nn.Module):
    """
    Creates smooth steps along x.

    Quite unstable in training, though.

    """
    def __init__(
            self,
            factor: float = 1.,
            steepness: float = 1.,
    ):
        super().__init__()
        assert factor > 0, f"Got {self._factor}"
        self._factor = factor
        self._steepness = steepness
        self._width = 25. * self._steepness
        self._hwidth = self._width / 2.
        self._factor2 = self._factor * self._width

    def extra_repr(self):
        return f"factor={self._factor}, steepness={self._steepness}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self._factor2 + 8
        return (F.sigmoid(y % self._width - self._hwidth) + torch.floor(y / self._width)) / self._factor

    @classmethod
    def create_from_string(cls, name: str) -> "SigmoidStep":
        """
        Create new instance

        :param name: str, can be
            - "sinstep"
            - "sinstep<factor>"
            - "sinstep<factor>,<steepness>"
        :return: SinStep instance
        """
        n = name.lower()
        if not n.startswith("sigmoidstep"):
            raise ValueError(f"Invalid name for SinStep '{name}'")
        n = n[11:]

        factor = ""
        for ch in n:
            if ch.isdigit() or ch == ".":
                factor = factor + ch
            else:
                break

        n = n[len(factor)+1:]
        return cls(
            factor=float(factor),
            steepness=float(n) if n else 1.,
        )
