from src.tests.base import *

import numpy as np
from scripts.extrusion_dataset.mesh import TriangleMesh


class TestMesh(TestBase):

    def test_100(self):
        mesh = TriangleMesh()
        mesh.add_triangle((0, 0, 0), (0, 1, 0), (1, 0, 0))
        self.assertNumpyEqual(
            np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype="f4"),
            mesh.vertices,
        )
        self.assertNumpyEqual(
            np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype="f4"),
            mesh.normals(),
        )

        mesh.add_triangle((0, 0, 0), (0, 0, 1), (0, 1, 0))
        self.assertNumpyEqual(
            np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]], dtype="f4"),
            mesh.normals(),
        )
