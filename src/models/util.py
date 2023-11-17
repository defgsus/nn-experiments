import itertools
from typing import List, Iterable, Tuple, Optional, Callable, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


def get_loss_callable(loss: Union[str, Callable, nn.Module]) -> Callable:
    if isinstance(loss, nn.Module) or callable(loss):
        return loss

    elif loss in ("l1", "mae"):
        return nn.L1Loss()

    elif loss in ("l2", "mse"):
        return nn.MSELoss()

    else:
        raise ValueError(f"Unexpected loss function '{loss}'")


class PrintLayer(nn.Module):
    def __init__(self, name: str = "debug"):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}:", x.shape if isinstance(x, torch.Tensor) else type(x).__name__)
        return x


class ResidualAdd(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class ResidualConcat(nn.Module):

    def __init__(self, module: nn.Module, dim: int = 1):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([x, self.module(x)], dim=self.dim)


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def extra_repr(self):
        return f"func={self.func}"


def activation_to_module(
        activation: Union[None, str, Callable, nn.Module, Type[nn.Module]]
) -> Union[None, nn.Module]:
    if activation is None:
        return None

    if isinstance(activation, nn.Module):
        return activation

    try:
        if issubclass(activation, nn.Module):
            return activation()
    except TypeError:
        pass

    if callable(activation):
        return Lambda(activation)

    if isinstance(activation, str):
        s = activation.lower()
        for module in (torch.nn, ):
            for key, value in vars(module).items():
                try:
                    if key.lower() == s and issubclass(value, nn.Module):
                        return value()
                except TypeError:
                    pass

    raise ValueError(f"Unrecognized activation: {repr(activation)}")
