import torch

from src.tests.base import TestBase
from src.transforms import *


class TestRandomCropTuple(TestBase):

    def test_100(self):
        cropper = RandomCropTuple(2)
        image = torch.Tensor([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ])
        cropped = cropper((image, image))
        self.assertTensorEqual(cropped[0], cropped[1])
