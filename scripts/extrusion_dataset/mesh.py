import math
from typing import Optional, Iterable, Tuple

import numpy as np
import pyrr.quaternion
from pyrr.vector3 import generate_normals
from pyrr import matrix33


class TriangleMesh:
    """
    A triangle mesh using numpy.

    With a large batch_size, a million triangles can be added per second
    """
    def __init__(self, vertices: Optional[Iterable] = None, batch_size: int = 100_000, dtype: str = "f4"):
        if vertices is not None:
            self._vertices = np.array(vertices).astype(dtype)
            if self._vertices.ndim == 1:
                self._vertices = self._vertices.reshape(-1, 3)
            if self._vertices.ndim != 2:
                raise ValueError(f"Expected [N*3] or [N, 3] shape, got {self._vertices.shape}")
        else:
            self._vertices = None

        self._dtype = dtype
        self._vertices_batch = []
        self._batch_size = batch_size

    def add_triangle(self, p1, p2, p3):
        assert len(p1) == 3
        assert len(p2) == 3
        assert len(p3) == 3
        self._vertices_batch.extend([*p1, *p2, *p3])
        if len(self._vertices_batch) > self._batch_size:
            self._flush_vertices()

    @property
    def vertices(self) -> np.ndarray:
        """
        Returns [V, 3] shape
        """
        self._flush_vertices()
        return self._vertices

    def normals(self) -> np.ndarray:
        vertices = self.vertices  # [V,3]
        normals = generate_normals(vertices[:-2:3], vertices[1:-1:3], vertices[2::3])  # [V/3,3]
        normals = normals.repeat(3, axis=0)  # [V,3]
        return normals

    def scaled(self, x, y, z) -> "TriangleMesh":
        v = self.vertices
        v = v * np.array([[x, y, z]], dtype=v.dtype)
        return TriangleMesh(v)

    def translated(self, x, y, z):
        v = self.vertices
        v = v + np.array([[x, y, z]], dtype=v.dtype)
        return TriangleMesh(v)

    def centered_at(self, x, y):
        cx, cy, _ = self.vertices.mean(axis=0)
        return self.translated(x - cx, y - cy, 0)

    def rotated_z(self, degree: float):
        return TriangleMesh(
            np.dot(
                self.vertices,
                matrix33.create_from_z_rotation(degree * math.pi / 180.),
            )
        )

    def _flush_vertices(self):
        if self._vertices_batch:
            new_vertices = np.array(self._vertices_batch, dtype=self._dtype).reshape(-1, 3)
            if self._vertices is None:
                self._vertices = new_vertices
            else:
                self._vertices = np.concatenate([self._vertices, new_vertices])
            self._vertices_batch.clear()

