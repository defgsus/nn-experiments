import math
import random
from typing import List, Iterable, Tuple, Optional, Callable, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.datasets import *
from src.algo.boulderdash import BoulderDash, BoulderDashGenerator


class BoulderDashIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            count: int,
            shape: Tuple[int, int] = (32, 32),
            one: float = 1.,
            zero: float = 0.,
            dtype: Optional[torch.dtype] = None,
            seed: Union[None, int] = None,
            generator_kwargs: Optional[Union[dict, Sequence[dict]]] = None,
            pre_steps: int = 0,
            prediction_steps: Optional[int] = None,
            include_bd: bool = False,
    ):
        self._shape = shape
        self._count = count
        self._seed = seed
        self._one = one
        self._zero = zero
        self._dtype = dtype
        self._pre_steps = pre_steps
        self._prediction_steps = prediction_steps
        self._include_bd = include_bd
        self._generator_kwargs = generator_kwargs

    def __len__(self):
        return self._count

    def __iter__(self):
        to_tensor_kwargs = dict(one=self._one, zero=self._zero, dtype=self._dtype)

        gen = BoulderDashGenerator(rng=self._seed)

        for i in range(self._count):

            if self._generator_kwargs is None:
                generator_kwargs = {}
            elif isinstance(self._generator_kwargs, dict):
                generator_kwargs = self._generator_kwargs
            else:
                generator_kwargs = gen.rng.choice(self._generator_kwargs)

            bd = gen.create_random(
                shape=self._shape,
                **generator_kwargs,
            )
            for i in range(self._pre_steps):
                bd.apply_physics()

            values = [
                bd.to_tensor(**to_tensor_kwargs)
            ]

            if self._prediction_steps:
                for i in range(self._prediction_steps):
                    bd.apply_physics()
                values.append(
                    bd.to_tensor(**to_tensor_kwargs)
                )

            if self._include_bd:
                values.append({"bd": bd})

            if len(values) == 1:
                yield values[0]
            else:
                yield tuple(values)

