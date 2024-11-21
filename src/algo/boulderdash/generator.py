import random
from typing import Union, Tuple, List

import numpy as np

from .boulderdash import BoulderDash
from src.algo import RandomVariable


class BoulderDashGenerator:

    def __init__(
            self,
            rng: Union[None, int, random.Random] = None
    ):
        if rng is None:
            self.rng = random
        elif isinstance(rng, int):
            self.rng = random.Random(rng)
        else:
            self.rng = rng

    def create_random(
            self,
            shape: Tuple[int, int],
            ratio_wall: Union[float, RandomVariable] = .15,
            ratio_wall_horizontal: Union[float, RandomVariable] = 0.,
            ratio_wall_horizontal_gap: Union[float, RandomVariable] = 0.3,
            ratio_rock: Union[float, RandomVariable] = 0.05,
            ratio_diamond: Union[float, RandomVariable] = .01,
            ratio_sand: Union[float, RandomVariable] = 0.2,
            with_border: bool = False,
    ) -> BoulderDash:

        def _get_value(x):
            if isinstance(x, RandomVariable):
                return x.get(self.rng)
            return x

        ratio_wall = _get_value(ratio_wall)
        ratio_wall_horizontal = _get_value(ratio_wall_horizontal)
        ratio_wall_horizontal_gap = _get_value(ratio_wall_horizontal_gap)
        ratio_rock = _get_value(ratio_rock)
        ratio_diamond = _get_value(ratio_diamond)
        ratio_sand = _get_value(ratio_sand)

        bd = BoulderDash(shape=shape)

        area = bd.shape[0] * bd.shape[1]
        if with_border:
            area = (bd.shape[0] - 2) * (bd.shape[1] - 2)
            self.draw_border(bd)

        if ratio_wall_horizontal > 0:
            self.draw_random_horizontally(
                bd,
                prob=ratio_wall_horizontal,
                prob_gap=ratio_wall_horizontal_gap,
            )

        coordinates = self._get_free_coordinates(bd)
        if not coordinates:
            raise RuntimeError(f"Failed to create a map with a player")
        self._place_random(bd, coordinates, bd.OBJECTS.Player)

        for ratio, obj in (
                (ratio_wall, bd.OBJECTS.Wall),
                (ratio_rock, bd.OBJECTS.Rock),
                (ratio_diamond, bd.OBJECTS.Diamond),
                (ratio_sand, bd.OBJECTS.Sand),
        ):
            if ratio > 0:
                for i in range(max(1, int(ratio * area))):
                    if not coordinates:
                        break
                    self._place_random(bd, coordinates, obj)
            if not coordinates:
                break

        return bd

    def draw_border(self, bd: BoulderDash, object: int = BoulderDash.OBJECTS.Wall):
        assert bd.shape[0] > 2 and bd.shape[1] > 2, f"Must have at least a 3x3 field to draw a border, got {bd.shape}"

        for y in range(bd.shape[0]):
            bd.map[y, 0] = [object, bd.STATES.Nothing]
            bd.map[y, bd.shape[1] - 1] = [object, bd.STATES.Nothing]

        for x in range(bd.shape[1]):
            bd.map[0, x] = [object, bd.STATES.Nothing]
            bd.map[bd.shape[0] - 1, x] = [object, bd.STATES.Nothing]

    def draw_random_horizontally(
            self,
            bd: BoulderDash,
            prob: float,
            prob_gap: float,
            object: int = BoulderDash.OBJECTS.Wall
    ):
        did_previously = False
        for y in range(bd.shape[0]):
            if self.rng.uniform(0, 1) < prob and not did_previously:
                did_previously = True
                for x in range(bd.shape[1]):
                    if self.rng.uniform(0, 1) >= prob_gap and bd.map[y, x, 0] == bd.OBJECTS.Empty:
                        bd.map[y, x] = [object, bd.STATES.Nothing]
                continue
            did_previously = False

    def _get_free_coordinates(self, bd: BoulderDash) -> List[List[int]]:
        return np.argwhere(bd.map[:, :, 0] == bd.OBJECTS.Empty).tolist()

    def _place_random(self, bd: BoulderDash, coordinates: List[List[int]], object: int) -> bool:
        if not coordinates:
            return False

        idx = self.rng.randrange(len(coordinates))
        y, x = coordinates.pop(idx)
        bd.map[y, x] = [object, bd.STATES.Nothing]
        return True
