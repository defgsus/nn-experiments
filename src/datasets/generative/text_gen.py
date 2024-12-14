import math
import random
from typing import Optional, Callable, List, Tuple, Iterable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from ..base_iterable import BaseIterableDataset


class TextMathIterableDataset(BaseIterableDataset):
    """
    Yields things like '3 + 4 = 7'
    """
    def __init__(
            self,
            count: Optional[int] = None,
            num_operands: int = 1,
            max_number: int = 10,
            operators: Iterable[str] = ("+",),
            sep: str = " ",
            fixed_width: Optional[int] = None,
            seed: Optional[int] = None,
            exclude: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self._count = count
        self._num_operands = num_operands
        self._max_number = max_number
        self._operators = list(operators)
        self._sep = sep
        self._fixed_width = fixed_width
        self._seed = seed
        self._exclude = None if exclude is None else set(exclude)
        if self._count is None:
            self._count = (max_number ** (num_operands + 1)) * (num_operands ** len(self._operators))

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Generator[str, None, None]:
        if self._seed is None:
            rng = random
        else:
            rng = random.Random(self._seed)

        num = 0
        while num < self._count:
            seq = [str(rng.randint(0, self._max_number))]
            for j in range(self._num_operands):
                seq.append(
                    rng.choice(self._operators)
                )
                seq.append(
                    str(rng.randint(0, self._max_number))
                )

            expression = self._sep.join(seq)
            result = str(eval(expression))
            expression = self._sep.join([expression, "=", result])

            if self._fixed_width:
                if self._fixed_width:
                    expression = expression.ljust(self._fixed_width)[:self._fixed_width]

            if self._exclude and expression in self._exclude:
                continue

            yield expression
            num += 1


class TextSelectiveCopyingIterableDataset(BaseIterableDataset):
    """
    Like described in the mamba paper https://arxiv.org/abs/2312.00752

    Yields things like 'A  BC D  : ABCD'
    """
    def __init__(
            self,
            count: int,
            num_items: int = 4,
            area: int = 10,
            seed: Optional[int] = None,
            exclude: Optional[Iterable[str]] = None,
            with_masked: bool = False,
    ):
        super().__init__()
        assert area >= num_items
        self._count = count
        self._num_items = num_items
        self._area = area
        self._seed = seed
        self._exclude = None if exclude is None else set(exclude)
        self._with_masked = with_masked

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Generator[str, None, None]:
        if self._seed is None:
            rng = random
        else:
            rng = random.Random(self._seed)

        num = 0
        while num < self._count:
            area = [" "] * self._area
            for item_idx in range(self._num_items):
                while True:
                    x = rng.randrange(len(area))
                    if area[x] == " ":
                        area[x] = chr(ord('A') + item_idx)
                        break

            question = "".join(area)
            result = question.replace(" ", "")

            expression = f"{question}: {result.ljust(self._num_items)}"

            if self._exclude and expression in self._exclude:
                continue

            if not self._with_masked:
                yield expression
            else:
                masked_expression = f"{question}: " + "\0" * self._num_items
                yield expression, masked_expression

            num += 1
