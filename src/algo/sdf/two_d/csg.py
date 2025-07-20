import numpy as np

from .base import *
from .util import mix


class ObjectGroup(Object):
    def __init__(
            self,
            *objects: Object,
    ):
        self.objects: Tuple[Object, ...] = objects


class Union(ObjectGroup):
    """Combines all objects"""
    def distance(self, space: np.ndarray) -> np.ndarray:
        d = None
        for obj in self.objects:
            obj_d = obj.distance(space)
            if d is None:
                d = obj_d
            else:
                d = np.minimum(d, obj_d)
        return d if d is not None else np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE


class Intersection(ObjectGroup):
    """Leaves intersection of first object and all following objects"""
    def distance(self, space: np.ndarray) -> np.ndarray:
        if not self.objects:
            return np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE

        d1 = self.objects[0].distance(space)
        d2 = Union(*self.objects[1:]).distance(space)

        return np.maximum(d1, d2)


class Subtraction(ObjectGroup):
    """
    Subtracts all following objects from the first object
    """
    def distance(self, space: np.ndarray) -> np.ndarray:
        if not self.objects:
            return np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE

        d1 = self.objects[0].distance(space)
        d2 = -Union(*self.objects[1:]).distance(space)

        return np.maximum(d1, d2)


class Xor(ObjectGroup):
    """
    XOR combination of all objects
    """
    def distance(self, space: np.ndarray) -> np.ndarray:
        if not self.objects:
            return np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE

        d1 = self.objects[0].distance(space)
        d2 = Xor(*self.objects[1:]).distance(space)

        return np.maximum(np.minimum(d1, d2), -np.maximum(d1, d2))


class SmoothUnion(ObjectGroup):
    """Combines all objects"""
    def __init__(
            self,
            *objects: Object,
            radius: float,
    ):
        super().__init__(*objects)
        self.radius = radius

    def distance(self, space: np.ndarray) -> np.ndarray:
        if not self.objects:
            return np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE

        d1 = self.objects[0].distance(space)
        d2 = SmoothUnion(*self.objects[1:], radius=self.radius).distance(space)

        h = np.clip(.5 + .5 * (d2 - d1) / max(EPS, self.radius), 0., 1.)
        return mix(d2, d1, h) - self.radius * h * (1. - h)


class SmoothSubtraction(ObjectGroup):
    """
    Subtracts all following objects from the first object
    """
    def __init__(
            self,
            *objects: Object,
            radius: float,
    ):
        super().__init__(*objects)
        self.radius = radius

    def distance(self, space: np.ndarray) -> np.ndarray:
        if not self.objects:
            return np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE

        d2 = self.objects[0].distance(space)
        # d2 = SmoothUnion(*self.objects[1:], radius=self.radius).distance(space)
        d1 = Union(*self.objects[1:]).distance(space)

        h = np.clip(.5 - .5 * (d2 + d1) / max(self.radius, EPS), 0., 1.)
        return mix(d2, -d1, h) - self.radius * h * (1. - h)


class SmoothIntersection(ObjectGroup):
    """Leaves intersection of first object and all following objects"""
    def __init__(
            self,
            *objects: Object,
            radius: float,
    ):
        super().__init__(*objects)
        self.radius = radius

    def distance(self, space: np.ndarray) -> np.ndarray:
        if not self.objects:
            return np.ones(space.shape[:2], dtype=DTYPE) * MAX_DISTANCE

        d1 = self.objects[0].distance(space)
        d2 = Union(*self.objects[1:]).distance(space)

        h = np.clip(.5 - .5 * (d2 - d1) / max(self.radius, EPS), 0., 1.)
        return mix(d2, d1, h) - self.radius * h * (1. - h)
