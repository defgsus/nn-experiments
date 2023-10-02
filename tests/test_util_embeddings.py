import math

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as VT

from tests.base import *
from src.util.embedding import *


class TestUtilEmbeddings(TestBase):

    def test_100_normalize(self):
        self.assertTrue(
            torch.all(
                (normalize_embedding(torch.randn(100, 10), save=False).pow(2).sum(1) - 1.).abs() < 0.001
            )
        )
        self.assertTrue(
            torch.all(
                (normalize_embedding(torch.randn(100, 10), save=True).pow(2).sum(1) - 1.).abs() < 0.001
            )
        )

    def test_200_normalize_save(self):
        self.assertTensorEqual(
            [
                [0.26726123690605164, 0.5345224738121033, 0.8017836809158325],
                [0, 0, 0],
                [0.7427813410758972, 0.5570860505104065, 0.3713906705379486],
            ],
            normalize_embedding(torch.Tensor([
                [1, 2, 3],
                [0, 0, 0],
                [4, 3, 2],
            ]))
        )
