import torch

from src.tests.base import *
from src.util import iter_batches


class TestUtilIterBatches(TestBase):

    def test_100_tensor(self):
        self.assertEqual(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9],
            ],
            [t.tolist() for t in iter_batches(torch.linspace(0, 9, 10), 3)]
        )

    def test_100_tuples(self):
        batches = list(iter_batches(
            [
                (torch.Tensor([1, 2]), 0),
                (torch.Tensor([3, 4]), 1),
                (torch.Tensor([5, 6]), 2),
                (torch.Tensor([7, 8]), 3),
            ],
            3
        ))

        self.assertEqual(
            [
                ([[1, 2], [3, 4], [5, 6]], [0, 1, 2]),
                ([[7, 8]], [3]),

            ],
            [(i[0].tolist(), i[1]) for i in batches]
        )
