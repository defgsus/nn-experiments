from .base import *


class Rounded(Object):

    def __init__(
            self,
            obj: Object,
            radius: float,
    ):
        self._obj = obj
        self._radius = radius

    def distance(self, space: np.ndarray) -> np.ndarray:
        return self._obj.distance(space) - self._radius


class SinWarp(Object):

    def __init__(
            self,
            obj: Object,
            amount: Vec2d,
            freq: Vec2d = 1.,
            phase: Vec2d = (0, 0),
    ):
        self._obj = obj
        self._freq = vec2_to_numpy(freq)
        self._phase = vec2_to_numpy(phase)
        self._amount = vec2_to_numpy(amount)

    def distance(self, space: np.ndarray) -> np.ndarray:
        warp = np.sin((space[..., ::-1] * self._freq + self._phase) * (np.pi * 2)) * self._amount
        space = space + warp
        return self._obj.distance(space)


class NoiseWarp(Object):

    def __init__(
            self,
            obj: Object,
            amount: Vec2d,
            freq: Union[int, Tuple[int, int]] = 1.,
            constant_edge: Optional[float] = None,
    ):
        self._obj = obj
        self._amount = vec2_to_numpy(amount)
        if isinstance(freq, int):
            freq = (freq, freq)
        self._freq = freq
        self._constant_edge = constant_edge

    def distance(self, space: np.ndarray) -> np.ndarray:
        warp = np.stack([
            perlin_noise_2d(space.shape[:2], self._freq, wrap=(True, True), constant_edge=self._constant_edge),
            perlin_noise_2d(space.shape[:2], self._freq, wrap=(True, True), constant_edge=self._constant_edge),
        ], axis=-1)
        space = space + warp * self._amount
        return self._obj.distance(space)


class NoiseWarpDistance(Object):

    def __init__(
            self,
            obj: Object,
            amount: float,
            freq: Union[int, Tuple[int, int]] = 1.,
            constant_edge: Optional[float] = None,
    ):
        self._obj = obj
        self._amount = amount
        if isinstance(freq, int):
            freq = (freq, freq)
        self._freq = freq
        self._constant_edge = constant_edge

    def distance(self, space: np.ndarray) -> np.ndarray:
        warp = perlin_noise_2d(space.shape[:2], self._freq, wrap=(True, True), constant_edge=self._constant_edge)
        d = self._obj.distance(space)
        return d + warp * self._amount
