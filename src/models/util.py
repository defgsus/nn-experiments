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
