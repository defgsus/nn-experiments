from typing import Union, Tuple, List, Optional, Iterable, Callable

import numpy as np
import pyrr.matrix33

from .util import (
    DTYPE, EPS, Vec2d, vec2_to_numpy,
    rotate_space, magnitude, perlin_noise_2d,
)

MAX_DISTANCE = 1_000_000_000_000.


class Object:

    def distance(self, space: np.ndarray) -> np.ndarray:
        """
        Calculate the distances for a 2d-array of (y, x) coordinates.

        :param space: numpy array of shape [H, W, 2]
        :return: numpy array of shape [H, W]
        """
        raise NotImplementedError

    def translate(self, translation: Vec2d):
        return Transform(self, translation=translation)

    def scale(self, scale: Union[Vec2d, float]):
        return Transform(self, scale=scale)

    def rotate(self, rotate_degree: float):
        return Transform(self, rotate_degree=rotate_degree)

    def round(self, radius: float):
        from .fx import Rounded
        return Rounded(self, radius=radius)

    def invert(self):
        return Transform(self, invert=True)

    def subtract(self, *objects: "Object", smooth: float = 0.):
        from .csg import Subtraction, SmoothSubtraction
        if smooth > 0:
            return SmoothSubtraction(self, *objects, radius=smooth)
        return Subtraction(self, *objects)

    def intersect(self, *objects: "Object", smooth: float = 0.):
        from .csg import Intersection, SmoothIntersection
        if smooth > 0:
            return SmoothIntersection(self, *objects, radius=smooth)
        return Intersection(self, *objects)

    def combine(self, *objects: "Object", smooth: float = 0.):
        from .csg import Union, SmoothUnion
        if smooth > 0:
            return SmoothUnion(self, *objects, radius=smooth)
        return Union(self, *objects)

    def sin_warp(
            self,
            amount: Vec2d,
            freq: Vec2d = 1.,
            phase: Vec2d = (0, 0),
    ):
        from .fx import SinWarp
        return SinWarp(self, amount=amount, freq=freq, phase=phase)

    def noise_warp(
            self,
            amount: Vec2d,
            freq: Union[int, Tuple[int, int]] = 1.,
            constant_edge: Optional[float] = None,
    ):
        from .fx import NoiseWarp
        return NoiseWarp(self, amount=amount, freq=freq, constant_edge=constant_edge)

    def noise_warp_distance(
            self,
            amount: float,
            freq: Union[int, Tuple[int, int]] = 1.,
            constant_edge: Optional[float] = None,
    ):
        from .fx import NoiseWarpDistance
        return NoiseWarpDistance(self, amount=amount, freq=freq, constant_edge=constant_edge)

    def render_distance(
            self,
            shape: Tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        space = np.mgrid[:shape[0], :shape[1]].transpose(1, 2, 0).astype(DTYPE)
        space /= vec2_to_numpy(shape)

        return self.distance(space)

    def render_mask(
            self,
            radius: Union[float, np.ndarray] = 0.,
            abs: bool = False,
            shape: Tuple[int, int] = (128, 128),
    ):
        d = self.render_distance(shape=shape)
        if abs:
            d = np.abs(d)
        if isinstance(radius, (int, float)):
            if radius == 0:
                inside = (d <= 0.).astype(DTYPE)
            elif radius > 0:
                inside = 1. - d.clip(0, radius) / radius
            else:
                inside = d.clip(radius, .0) / radius
        else:
            if radius.ndim != 2:
                raise ValueError(f"Expected radius shape [H, W], got {radius.shape}")

            # flatten for boolean indexing
            radius = radius.flatten()
            d = d.flatten()
            rad_zero = radius == 0
            rad_neg = radius < 0
            rad_pos = radius > 0
            inside = np.zeros((shape[0] * shape[1]), dtype=DTYPE)
            inside[rad_zero] = d[rad_zero] <= 0
            inside[rad_pos] = 1. - (d[rad_pos] / radius[rad_pos]).clip(0, 1)
            inside[rad_neg] = (d[rad_neg] / radius[rad_neg]).clip(0, 1)
            inside = inside.reshape(shape)
        return inside


    def render_distance_normal(
            self,
            eps: Optional[float] = None,
            shape: Tuple[int, int] = (128, 128),
    ):
        if eps is None:
            eps = 1. / min(shape)
        n = np.stack([
            self.translate((eps, 0)).render_distance(shape) - self.translate((-eps, 0)).render_distance(shape),
            self.translate((0, eps)).render_distance(shape) - self.translate((0, -eps)).render_distance(shape),
            ], axis=-1)
        m = magnitude(n)
        mask = m > 0
        n[mask] = n[mask] / m[mask][..., None]
        return n

    def render_mask_normal(
            self,
            radius: float = .0,
            abs: bool = False,
            eps: Optional[float] = None,
            shape: Tuple[int, int] = (128, 128),
            mask_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            z: Optional[float] = None,
    ):
        def _render(obj):
            m = obj.render_mask(radius=radius, shape=shape, abs=abs)
            if mask_func is not None:
                m = mask_func(m)
            return m

        if eps is None:
            eps = 1. / min(shape)
        n = [
            (_render(self.translate((eps, 0))) - _render(self.translate((-eps, 0)))) / eps,
            (_render(self.translate((0, eps))) - _render(self.translate((0, -eps)))) / eps,
        ]
        if z is not None:
            n.append(np.ones_like(n[0]) * z)
        n = np.stack(n, axis=-1)
        m = magnitude(n)
        mask = m > 0
        n[mask] /= m[mask][..., None]
        return n


class Transform(Object):
    def __init__(
            self,
            obj: Object,
            translation: Optional[Vec2d] = None,
            scale: Optional[float] = None,
            rotate_degree: Optional[float] = None,
            invert: bool = False,
    ):
        self._obj = obj
        self._translation = translation
        self._scale = scale
        self._rotate_degree = rotate_degree
        self._invert = invert

    def transform(self, space: np.ndarray) -> np.ndarray:
        if self._translation is not None:
            space = space - vec2_to_numpy(self._translation)
        if self._scale is not None:
            space = space / self._scale
        if self._rotate_degree is not None:
            space = rotate_space(space, self._rotate_degree)

        return space

    def transform_distance(self, d: np.ndarray) -> np.ndarray:
        if self._scale is not None:
            d = d * self._scale
        if self._invert:
            d = -d
        return d

    def distance(self, space: np.ndarray) -> np.ndarray:
        return self.transform_distance(self._obj.distance(self.transform(space)))

