import itertools
from typing import Tuple, Union, Optional, Iterable, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class TotalCADataset(Dataset):

    def __init__(
            self,
            shape: Union[torch.Size, Tuple[int, int]],
            num_iterations: int = 10,
            init_prob: float = .5,
            wrap: bool = False,
            dtype: torch.dtype = torch.float,

    ):
        # assert padding_mode in ("zeros", "reflect", "replicate")

        self.shape = torch.Size(shape)
        self.num_iterations = num_iterations
        self.init_prob = init_prob
        self.wrap = wrap
        self.dtype = dtype

        #self.conv = nn.Conv2d(
        #    1, 1,
        #    kernel_size=3,
        #    padding_mode="circular" if self.wrap else "zeros",
        #    padding=1,
        #    bias=False,
        #)
        # 1x1x3x3
        self.kernel = torch.Tensor([[[
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]]]).to(self.dtype)#, requires_grad=False)

    def __len__(self):
        return 2 ** 18

    def __getitem__(self, item: Union[int, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(item, str):
            index = self.rule_to_index(item)
        else:
            index = item

        birth, survive = self.index_to_rule(index)

        cells = self.init_cells()
        for iter in range(self.num_iterations):
            cells = self.step_cells(cells, birth, survive)

        return (
            cells,
            torch.Tensor([(index >> b) & 1 for b in range(18)]).to(self.dtype)
        )

    def index_to_rule(
            self,
            index: int,
    ) -> Tuple[List[int], List[int]]:
        r1 = index & (2 ** 9 - 1)
        r2 = (index >> 9) & (2 ** 9 - 1)
        birth = [b for b in range(9) if (r1 >> b) & 1]
        survive = [b for b in range(9) if (r2 >> b) & 1]
        return birth, survive

    def rule_to_index(
            self,
            rule: Union[str, Tuple[Iterable[int], Iterable[int]]] = "3-23"
    ):
        if isinstance(rule, str):
            r1, r2 = rule.split("-")
        else:
            r1, r2 = rule

        index = 0
        for n in r1:
            index |= (1 << int(n))
        for n in r2:
            index |= (1 << (int(n) + 9))
        return index

    def init_cells(self) -> torch.Tensor:
        return (torch.rand(*self.shape) <= self.init_prob).to(self.dtype)

    def step_cells(self, cells: torch.Tensor, birth: Iterable[int], survive: Iterable[int]) -> torch.Tensor:
        neigh = self.total_neighbours(cells)
        dead = cells == 0
        alive = torch.logical_not(dead)
        new_state = torch.zeros_like(cells, dtype=torch.bool)

        for num_n in birth:
            new_state |= dead & (neigh == num_n)
        for num_n in survive:
            new_state |= alive & (neigh == num_n)

        return new_state.to(self.dtype)

    def total_neighbours(self, cells: torch.Tensor) -> torch.Tensor:
        if not self.wrap:
            return F.conv2d(
                input=cells.unsqueeze(0),
                weight=self.kernel,
                padding=1,  #self.padding_mode,
            )[0]
        else:
            # Note: torch does only support 2-tuple for "circular"
            cells_padded = F.pad(cells.unsqueeze(0), (1, 1), mode="circular")
            #   ... so the y wrapping is done with the transposed cells
            cells_padded = F.pad(cells_padded.permute(0, 2, 1), (1, 1), mode="circular").permute(0, 2, 1)

            return F.conv2d(
                input=cells_padded,
                weight=self.kernel,
            )[0]
