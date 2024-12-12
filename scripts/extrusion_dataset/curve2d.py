import math
from typing import List, Tuple, Union, Callable, Generator

from .mesh import TriangleMesh


class Curve2dBase:

    def __init__(self):
        pass

    def max_t(self) -> float:
        raise NotImplementedError

    def __call__(self, global_t: float):
        raise NotImplementedError

    def derivative(self, global_t: float, eps: float = 0.001, normalized: bool = False):
        p1 = self(global_t - eps)
        p2 = self(global_t + eps)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if normalized:
            mag = math.sqrt(dx*dx + dy*dy)
            if mag:
                dx, dy = dx / mag, dy / mag
        return dx, dy

    def iter_points(
            self,
            point_distance: float,
            threshold: float = 0.0001,
            max_tries: int = 100,
            eps: float = 0.0001,
    ) -> Generator[Tuple[int, int], None, None]:
        assert max_tries
        t = 0.
        p1 = self(t)
        step = point_distance / 3
        while t < self.max_t():

            step_fac = distance(p1, self(t + 0.1)) / 0.1
            for _ in range(max_tries):
                p2 = self(t + step)
                dist = distance(p1, p2)

                if abs(point_distance - dist) < threshold:
                    break

                step = max(eps, step + .1 / step_fac * (point_distance - dist))

            yield p1
            p1 = p2
            t += step

    def iter_edges(
            self,
            radius: Union[float, Callable[[float], float]],
            point_distance: float,
            threshold: float = 0.0001,
            max_tries: int = 100,
            eps: float = 0.0001,
    ) -> Generator[Tuple[float, Tuple[int, int], Tuple[int, int], Tuple[int, int]], None, None]:
        t = 0.
        p1 = self(t)
        d = self.derivative(t, normalized=True)
        r = radius(t / self.max_t()) if callable(radius) else radius
        d = (d[0] * r, d[1] * r)
        p1l = (p1[0] - d[1], p1[1] + d[0])
        p1r = (p1[0] + d[1], p1[1] - d[0])
        step = point_distance / 3
        while t < self.max_t():

            step_fac = distance(p1, self(t + 0.1)) / 0.1

            r = radius(t / self.max_t()) if callable(radius) else radius

            for _ in range(max_tries):
                p2 = self(t + step)

                d = self.derivative(t + step, normalized=True)
                d = (d[0] * r, d[1] * r)
                p2l = (p2[0] - d[1], p2[1] + d[0])
                p2r = (p2[0] + d[1], p2[1] - d[0])

                dist = max(
                    distance(p1l, p2l),
                    distance(p1r, p2r),
                )

                if abs(point_distance - dist) < threshold:
                    break

                step = max(eps, step + .1 / step_fac * (point_distance - dist))

            yield t, r, p2, p2l, p2r
            p1 = p2
            p1l, p1r = p2l, p2r
            t += step

    def to_mesh(
            self,
            radius: Union[float, Callable[[float], float]],
            point_distance: float,
    ) -> TriangleMesh:

        mesh = TriangleMesh()
        m1, l1, r1 = None, None, None
        for t, r, m2, l2, r2 in self.iter_edges(radius, point_distance):
            #t = t / self.max_t()
            ra = min(t * 50, 1, max(0, (self.max_t() - t) * 50))
            l2 = (*l2, 0)
            r2 = (*r2, 0)
            m2 = (*m2, r * ra)
            if m1 is not None:

                mesh.add_triangle(l1, m1, l2)
                mesh.add_triangle(m1, m2, l2)

                mesh.add_triangle(r1, r2, m1)
                mesh.add_triangle(m1, r2, m2)

            m1, l1, r1 = m2, l2, r2

        return mesh


class CurveCircle2d(Curve2dBase):
    def __init__(
            self,
            radius: float,
            range: Tuple[float, float] = (0, 1),
    ):
        super().__init__()
        self.radius = radius
        self.range = range

    def max_t(self) -> float:
        return self.range[1] - self.range[0]

    def __call__(self, global_t: float):
        t = (global_t - self.range[0]) * math.pi * 2
        return (
            math.sin(t) * self.radius,
            math.cos(t) * self.radius,
        )


class CurveLinear2d(Curve2dBase):
    """
    a variation of the hermite interpolation
    https://paulbourke.net/miscellaneous/interpolation/
    """
    def __init__(
            self,
            points: List[Tuple[float, float]],
    ):
        super().__init__()
        self.points = points

    def max_t(self) -> float:
        return len(self.points) - 1

    def _get_point(self, i: int):
        assert len(self.points) >= 2
        if i < 0:
            return (2 * self.points[0][0] - self.points[1][0], 2 * self.points[0][1] - self.points[1][1])
        if i >= len(self.points):
            return (2 * self.points[-1][0] - self.points[-2][0], 2 * self.points[-1][1] - self.points[-2][1])
        return self.points[i]

    def __call__(self, global_t: float):
        ti = int(global_t)

        p1 = self._get_point(ti)
        p2 = self._get_point(ti + 1)

        t = global_t - ti
        return (
            p1[0] * (1. - t) + t * p2[0],
            p1[1] * (1. - t) + t * p2[1],
        )


class CurveHermite2d(CurveLinear2d):

    def __call__(self, global_t: float):
        ti = int(global_t)

        p0 = self._get_point(ti - 1)
        p1 = self._get_point(ti)
        p2 = self._get_point(ti + 1)

        d1 = (p1[0] - p0[0], p1[1] - p0[1])
        d2 = (p2[0] - p1[0], p2[1] - p1[1])

        t = global_t - ti
        tsq2 = t * t
        tsq3 = tsq2 * t
        f1 = 2*tsq3 - 3*tsq2 + 1
        f2 = tsq3 - 2*tsq2 + t

        d = (d1[0] * (1. - t) + t * d2[0], d1[1] * (1. - t) + t * d2[1])

        return (
            p1[0] * f1 + d[0] * f2 + (tsq3 - tsq2) * d[0] + (1. - f1) * p2[0],
            p1[1] * f1 + d[1] * f2 + (tsq3 - tsq2) * d[1] + (1. - f1) * p2[1],
        )

        # f = t*t*(3 - 2 * t)
        return (
            p1[0] * (1. - f) + f * p2[0],
            p1[1] * (1. - f) + f * p2[1],
        )


def distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    #return abs(dx) + abs(dy)
    return math.sqrt(dx*dx + dy*dy)

