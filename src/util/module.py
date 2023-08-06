import math

import torch
import torch.nn as nn


def num_module_parameters(module: nn.Module) -> int:
    count = 0
    for p in module.parameters():
        count += math.prod(p.shape)
    return count
