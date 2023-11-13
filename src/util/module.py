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