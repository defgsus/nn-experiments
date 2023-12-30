import torch

from tests.base import *

from src.algo.ca1 import util


class TestUtil(TestBase):

    def test_100_pad(self):
        self.assertTensorEqual(
            [0, 1, 2, 0],
            util.pad(torch.Tensor([1, 2]), (1, 1), wrap=False)
        )
        self.assertTensorEqual(
            [[0, 1, 2, 0]],
            util.pad(torch.Tensor([[1, 2]]), (1, 1), wrap=False)
        )
        self.assertTensorEqual(
            [2, 1, 2, 1],
            util.pad(torch.Tensor([1, 2]), (1, 1), wrap=True)
        )
        self.assertTensorEqual(
            [1, 2, 1, 2, 1, 2, 1],
            util.pad(torch.Tensor([1, 2]), (0, 5), wrap=True)
        )
        self.assertTensorEqual(
            [2, 1, 2, 1, 2],
            util.pad(torch.Tensor([1, 2]), (3, 0), wrap=True)
        )
        self.assertTensorEqual(
            [2, 1, 2, 1, 2, 1, 2, 1, 2],
            util.pad(torch.Tensor([1, 2]), (3, 4), wrap=True)
        )

    def test_110_pad_fuzzing(self):
        for length in range(1, 11):
            for pad_x in range(0, 30):
                for pad_y in range(0, 30):
                    for wrap in (False, True):

                        input = torch.zeros(length)
                        for i in range(2):
                            padded = util.pad(input, (pad_x, pad_y), wrap=wrap)
                            self.assertEqual(length + pad_x + pad_y, padded.shape[-1])

                            input = torch.zeros(5, length)
