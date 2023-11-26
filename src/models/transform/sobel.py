import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel(nn.Module):
    def __init__(
            self,
            magnitude: bool = True,
            direction: bool = True,
            padding: int = 0,
    ):
        super().__init__()
        self.magnitude = magnitude
        self.direction = direction
        self.padding = padding
        self.kernel_1 = nn.Parameter(torch.Tensor([[[
            [1, 0, -1], [2, 0, -2], [1, 0, -1]
        ]]]), requires_grad=False)
        self.kernel_2 = nn.Parameter(torch.Tensor([[[
            [1, 2, 1], [0, 0, 0], [-1, -2, -1]
        ]]]), requires_grad=False)

    def forward(self, x):
        g1 = F.conv2d(x, self.kernel_1, padding=self.padding)
        g2 = F.conv2d(x, self.kernel_2, padding=self.padding)

        if self.magnitude:
            mag = torch.sqrt(g1 ** 2 + g2 ** 2)

        if self.direction:
            dir = torch.atan2(g1, g2)

        if self.magnitude:
            if not self.direction:
                return mag
            else:
                return torch.concat([mag, dir], dim=1)
        else:
            if self.direction:
                return dir
            else:
                raise ValueError("Must define at least one of `magnitude` or `direction`")
