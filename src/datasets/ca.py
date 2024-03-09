import itertools
from typing import Tuple, Union, Optional, Iterable, List, Callable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.base_dataset import BaseDataset


class TotalCADataset(BaseDataset):

    def __init__(
            self,
            shape: Union[torch.Size, Tuple[int, int]],
            num_iterations: Union[int, Tuple[int, int]] = 10,
            init_prob: Union[float, Tuple[float, float]] = .5,
            seed: Optional[int] = None,
            num_repetitions: int = 1,
            wrap: bool = False,
            rules: Optional[Iterable[Union[int, str]]] = None,
            dtype: torch.dtype = torch.uint8,
            transforms: Optional[List[Union[nn.Module, Callable]]] = None,
    ):
        assert num_repetitions >= 1

        self.shape = torch.Size(shape)
        self.num_iterations = num_iterations
        self.num_repetitions = num_repetitions
        self.init_prob = init_prob
        self.seed = seed
        self.wrap = wrap
        self.dtype = dtype
        self.transforms = transforms
        self.rules = None
        if rules is not None:
            self.rules = [
                r if isinstance(r, int) else self.rule_to_number(r)
                for r in rules
            ]

        # 1x1x3x3
        self.kernel = torch.Tensor([[[
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]]]).to(self.dtype)

    def __len__(self):
        if self.rules is not None:
            return len(self.rules) * self.num_repetitions
        return (2 ** 18) * self.num_repetitions

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitem__(self, item: Union[int, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(item, str):
            index = self.rule_to_index(item)
        else:
            index = item

        rng = torch.default_generator
        if self.seed is not None:
            rng = torch.Generator().manual_seed(self.seed)
            rng = torch.Generator().manual_seed(index + torch.randint(1, 2**60, [1], generator=rng).item())

        birth, survive = self.index_to_rule(index)
        num_iterations = self.num_iterations
        if isinstance(num_iterations, (list, tuple)):
            if self.num_iterations[0] == self.num_iterations[1]:
                num_iterations = self.num_iterations[0]
            else:
                num_iterations = torch.randint(
                    self.num_iterations[0],
                    self.num_iterations[1],
                    (1,), generator=rng
                )[0]

        cells = self.init_cells(rng)
        for iter in range(num_iterations):
            cells = self.step_cells(cells, birth, survive)

        if self.transforms is not None:
            for t in self.transforms:
                cells = t(cells)

        return (
            cells,
            torch.Tensor([(index >> b) & 1 for b in range(18)]).to(self.dtype)
        )

    def index_to_rule(
            self,
            index: int,
    ) -> Tuple[List[int], List[int]]:
        index //= self.num_repetitions

        if self.rules is not None:
            index = self.rules[index]

        r1 = index & (2 ** 9 - 1)
        r2 = (index >> 9) & (2 ** 9 - 1)
        birth = [b for b in range(9) if (r1 >> b) & 1]
        survive = [b for b in range(9) if (r2 >> b) & 1]
        return birth, survive

    def rule_to_index(
            self,
            rule: Union[str, Tuple[Iterable[int], Iterable[int]]] = "3-23"
    ) -> int:
        if self.rules is not None:
            raise NotImplementedError(f"Can't use rule_to_index on dataset with limited rules")
        return self.rule_to_number(rule)

    @classmethod
    def rule_to_number(
            cls,
            rule: Union[str, Tuple[Iterable[int], Iterable[int]]] = "3-23"
    ) -> int:
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

    def init_cells(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        init_prob = (
            torch.rand(1, generator=rng)[0] * (self.init_prob[1] - self.init_prob[0]) + self.init_prob[0]
            if isinstance(self.init_prob, (list, tuple)) else self.init_prob
        )
        return (torch.rand(*self.shape, generator=rng) < init_prob).to(self.dtype)

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
