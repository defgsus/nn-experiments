import unittest

import torch
import torch.nn.functional as F

from src.models.recursive import RecursiveUnit


class TestModelRecursive(unittest.TestCase):

    def test_100_unit(self):
        for n_cells in (1, 10, 1000):
            unit = RecursiveUnit(n_cells=n_cells, init_weights=True)

            with torch.no_grad():
                w_min, w_max = float(unit.recursive_weights.min()), float(unit.recursive_weights.max())

            for input_shape in (
                    (n_cells,),
                    (1, n_cells),
                    (16, n_cells),
            ):
                x = torch.rand(*input_shape) * 2. - 1.

                y = unit.forward(x, n_iter=10)

                self.assertEqual(input_shape, y.shape)

                with torch.no_grad():
                    y_min, y_max = float(y.min()), float(y.max())

                print(f"shape {str(input_shape):14} out-range {y_min:3.4f} - {y_max:3.4f}")

