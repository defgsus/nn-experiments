import math
from typing import Dict

import numpy as np

from .boulderdash import BoulderDash


class BoulderDashGraphics:
    """
    Some lazy-live-rendered graphics of variable size for displaying BoulderDash maps
    """

    _graphics: Dict[int, Dict[int, np.ndarray]] = {}

    @classmethod
    def graphic(cls, object: int, size: int = 8):
        if size not in cls._graphics:
            cls._graphics[size] = {}
        if object not in cls._graphics[size]:
            cls._graphics[size][object] = cls._render(object, size)
        return cls._graphics[size][object]

    @classmethod
    def _render(cls, object: int, size: int):
        if object == BoulderDash.OBJECTS.Wall:
            b = max(1, int(size / 8))
            g = np.ones((3, size, size)) * .5
            g[:, :b, :] *= 1.5
            g[:, :, :b] *= 1.5
            g[:, -b:, :] *= .7
            g[:, :, -b:] *= .7
            return np.clip(g, 0, 1)

        elif object == BoulderDash.OBJECTS.Rock:
            yx = np.mgrid[:size, :size] / size * 2. - 1.
            d = np.sqrt(np.square(yx[0]) + np.square(yx[1]))
            form = (d < 1.).astype(np.float_)
            g = form
            b = max(1, int(size / 12))
            g[:-b, :-b] += .2 * form[b:, b:]
            g[b:, b:] -= .2 * form[:-b, :-b]
            g = g[None, :, :].repeat(3, 0)
            return np.clip(g * .6, 0, 1)

        elif object == BoulderDash.OBJECTS.Sand:
            yx = np.mgrid[:size, :size] / size * math.pi * 4
            g = .9 + .1 * np.sin(yx[0]) * np.sin(yx[1])
            g = g[None, :, :].repeat(3, 0)
            g[0] *= .5
            g[1] *= .3
            g[2] *= .1
            return g

        elif object == BoulderDash.OBJECTS.Diamond:
            yx = np.mgrid[:size, :size] / size * 2. - 1.
            d = np.abs(yx[0]) + np.abs(yx[1])
            form = (d < 1.).astype(float)
            g = form
            b = max(1, int(size / 12))
            g[:-b, :-b] += .2 * form[b:, b:]
            g[b:, b:] -= .2 * form[:-b, :-b]
            g = g[None, :, :].repeat(3, 0)
            g[1] *= 1.3
            g[2] *= 1.7
            return np.clip(g * .6, 0, 1)

        elif object == BoulderDash.OBJECTS.Player:
            yx = np.mgrid[:size, :size] / size * 2. - 1.
            yx[0] = yx[0] * 1.5
            yx[1] *= 2.

            d = np.sqrt(np.square(yx[0]) + np.square(yx[1]))
            form = (d < 1.).astype(np.float_)
            g = form
            b = max(1, int(size / 12))
            g[:-b, :-b] += .2 * form[b:, b:]
            g[b:, b:] -= .2 * form[:-b, :-b]
            g = g[None, :, :].repeat(3, 0)
            g[1] *= 1.5
            return np.clip(g * .6, 0, 1)

        else:  # object == BoulderDash.OBJECTS.Empty:
            return np.zeros((3, size, size))
