from .base import *


class Plane(Object):
    def __init__(
            self,
            normal: Vec2d = (1, 0),
    ):
        self.normal = normal

    def distance(self, space: np.ndarray) -> np.ndarray:
        d = space * vec2_to_numpy(self.normal)
        return d.sum(axis=-1)


class Circle(Object):
    def __init__(
            self,
            radius: float,
    ):
        self.radius = radius

    def distance(self, space: np.ndarray) -> np.ndarray:
        return magnitude(space) - self.radius


class Box(Object):
    def __init__(
            self,
            radius: Vec2d = (1, 0),
    ):
        self.radius = radius

    def distance(self, space: np.ndarray) -> np.ndarray:
        v = np.abs(space) - vec2_to_numpy(self.radius)
        return magnitude(np.maximum(v, 0.)) + np.minimum(np.maximum(v[..., 0], v[..., 1]), 0.)
