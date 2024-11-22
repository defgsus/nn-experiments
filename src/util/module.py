import math
from typing import Optional

import torch
import torch.nn as nn


def num_module_parameters(module: nn.Module, trainable: Optional[bool] = None) -> int:
    count = 0
    for p in module.parameters():
        if trainable is None or trainable == p.requires_grad:
            count += p.numel()
    return count


def clip_module_weights(module: nn.Module, max_magnitude: float):
    with torch.no_grad():
        for param in module.parameters():
            param[:] = param.clamp(-max_magnitude, max_magnitude)
            #print(param.max(), param.shape)
