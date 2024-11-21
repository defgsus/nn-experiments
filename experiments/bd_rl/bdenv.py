import random
import math
from typing import Optional, Tuple, Union

import torch

from src.algo.boulderdash import *


class BoulderDashEnvironment:

    REWARDS = {
        BoulderDash.RESULTS.Nothing: 0.0,
        BoulderDash.RESULTS.Blocked: -0.1,
        BoulderDash.RESULTS.Moved: 0.01,
        BoulderDash.RESULTS.RemovedSand: 0.01,
        BoulderDash.RESULTS.PushedRock: 0.01,
        BoulderDash.RESULTS.PlayerDied: -1.0,
        BoulderDash.RESULTS.CollectedDiamond: 1.,
        BoulderDash.RESULTS.CollectedAllDiamonds: 10.,
    }

    def __init__(self, shape: Tuple[int, int] = (16, 16)):
        self.shape = shape
        self.bd: Optional[BoulderDash] = None
        self.all_actions = [i[1] for i in BoulderDash.ACTIONS.items()]
        self._stack = []
        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.bd = BoulderDashGenerator(rng=seed).create_random(
            shape=self.shape
        )
        if not self.bd.player_position():
            raise RuntimeError(f"Got a map without a player:\n{self.bd.to_string_map()}")

    def act(self, action: Union[int, torch.Tensor]) -> Tuple[int, float, bool]:
        if isinstance(action, torch.Tensor):
            action = action.argmax()

        result1 = self.bd.apply_action(action)
        result2 = self.bd.apply_physics()

        result = result1 if result2 == self.bd.RESULTS.Nothing else result2

        terminated = result in (
            self.bd.RESULTS.CollectedAllDiamonds,
            self.bd.RESULTS.PlayerDied,
        )
        reward = self.REWARDS[result]

        return result, reward, terminated

    def get_random_action(self) -> int:
        return random.choice(self.all_actions)

    def state(self):
        return self.bd.to_tensor(one=1, zero=-1)

    def push_state(self):
        self._stack.append(self.bd.to_array())

    def pop_state(self):
        self.bd = BoulderDash.from_array(self._stack.pop())
