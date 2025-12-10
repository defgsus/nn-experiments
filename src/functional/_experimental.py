import torch
import torch.nn as nn
import torch.nn.functional as F


class Poly3(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clamp(0, 1.5) #67765)
        y = y * y * (3. - 2. * y)
        return y
