from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalCALayer(nn.Module):
    def __init__(
            self,
            birth: Union[None, torch.Tensor, Iterable[int]] = None,
            survive: Union[None, torch.Tensor, Iterable[int]] = None,
            iterations: int = 1,
            learn_kernel: bool = False,
            learn_rules: bool = False,
            wrap: bool = False,
            threshold: float = .5,
            alpha: float = 1.,
    ):
        """
        Totalitarian Cellular Automaton as torch layer.
        
        :param birth: optional birth rule
        :param survive: optional survival rule 
        :param iterations: number of iterations
        :param learn_kernel: do train the neighbourhood kernel 
        :param learn_rules: do train the rules
        :param wrap: if True, edges wrap around
        :param threshold: float threshold on which a cell is considered alive
        :param alpha: float mix of the calculation result [0, 1]
        """
        super().__init__()
        for name, value in (("birth", birth), ("survive", survive)):
            if value is None:
                value = torch.rand(9).bernoulli()
            elif not isinstance(value, torch.Tensor):
                value = torch.Tensor(value)

            if value.shape != torch.Size((9, )):
                raise ValueError(f"Expected `{name}` to have shape (9), got {value.shape}")

            setattr(self, name, nn.Parameter(value, requires_grad=learn_rules))

        self.iterations = iterations
        self.wrap = wrap
        self.threshold = threshold
        self.kernel = nn.Parameter(torch.Tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]]), requires_grad=learn_kernel)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        if ndim not in (3, 4):
            raise ValueError(f"Expected x.ndim == 3 or 4, got {x.shape}")
        if ndim == 3:
            x = x.unsqueeze(0)

        y = x
        ch = x.shape[-3]
        if ch != 1:
            y = y.view(-1, 1, *x.shape[-2:])  # (BxC)xHxW

        for i in range(self.iterations):
            y = self._ca_step(y)

        if ch != 1:
            y = y.view(x.shape)

        if self.alpha != 1.:
            y = x * (1. - self.alpha) + self.alpha * y

        return y if ndim == 4 else y.squeeze(0)

    def _ca_step(self, x: torch.Tensor) -> torch.Tensor:

        cells = (x >= self.threshold).float()

        if self.wrap:
            cellsp = torch.concat([cells[..., -1, None], cells, cells[..., 0, None]], dim=-1)
            cellsp = torch.concat([cellsp[..., -1, None, :], cellsp, cellsp[..., 0, None, :]], dim=-2)
            neighbour_count = F.conv2d(cellsp, self.kernel)
        else:
            neighbour_count = F.conv2d(cells, self.kernel, padding=1)

        neighbour_count = neighbour_count.long().clamp(0, 8)

        birth = torch.index_select(self.birth, 0, neighbour_count.flatten(0)).view(x.shape)
        survive = torch.index_select(self.survive, 0, neighbour_count.flatten(0)).view(x.shape)

        return birth * (1. - cells) + survive * cells
