import time
import unittest
import math
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.datasets import ContrastiveIterableDataset


class TestDatasetContrastiveIterable(unittest.TestCase):

    def test_100(self):
        import torchvision.models
        source_ds = TensorDataset(
            torch.linspace(0, 199, 200).view(100, 2),
            torch.linspace(0, 49, 50).repeat(2),
        )
        print(source_ds[0][0].shape, source_ds[0][1].shape)
        ds = ContrastiveIterableDataset(
            source_ds,
            contrastive_ratio=.5,
        )
        #b1, b2, is_same \
        x = DataLoader(ds, batch_size=100)
        print(x)
        print(ds._item_map)