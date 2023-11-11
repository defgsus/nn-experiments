from typing import List, Iterable, Tuple, Optional, Callable, Union

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
