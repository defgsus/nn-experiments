import random
import math
from typing import Optional, Union, Tuple, Type, Sequence, Any


class RandomVariable:

    def __init__(
            self,
            type: Optional[Type] = None
    ):
        self.type = type

    def get(self, rng: Optional[random.Random] = None):
        if rng is None:
            rng = random
        v = self._get(rng)
        if self.type is not None:
            v = type(v)
        return v

    def _get(self, rng: random.Random):
        raise NotImplementedError


class RandomRangeVariable(RandomVariable):
    """Generates values in the range of [min, max]"""
    def __init__(
            self,
            min: float,
            max: float,
            power: float = 1.,
            type: Optional[Type] = None,
    ):
        super().__init__(type)
        self.min = min
        self.max = max
        self.power = power

    def _get(self, rng: random.Random):
        t = rng.uniform(0, 1)
        if self.min == self.max:
            return self.min
        t = math.pow(t, self.power)
        return self.min + t * (self.max - self.min)


class RandomChoiceVariable(RandomVariable):
    def __init__(
            self,
            choices: Sequence[Any],
            weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(type)
        self.choices = list(choices)
        self.weights = None if weights is None else list(weights)

    def _get(self, rng: random.Random):
        return rng.choices(self.choices, self.weights)[0]
