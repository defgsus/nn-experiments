import time
import unittest
import math
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader
from src.datasets import ClassFeaturesDataset


class TestClassFeaturesDataset(unittest.TestCase):

    def test_100_pure(self):
        ds = TensorDataset(torch.Tensor([
            23, 42, 66, 42, 66,
        ]).to(torch.int32))
        ds = ClassFeaturesDataset(ds, dtype=torch.float16)
        self.assertEqual(
            [
                ([1.0, 0.0, 0.0], 23),
                ([0.0, 1.0, 0.0], 42),
                ([0.0, 0.0, 1.0], 66),
                ([0.0, 1.0, 0.0], 42),
                ([0.0, 0.0, 1.0], 66),
            ],
            [
                (feature.tolist(), label.item())
                for feature, label in ds
            ]
        )

    def test_110_pure_fixed_num(self):
        ds = TensorDataset(torch.Tensor([
            23, 42, 66, 42, 66,
        ]).to(torch.int32))
        ds = ClassFeaturesDataset(ds, num_classes=5)
        self.assertEqual(
            [
                ([1.0, 0.0, 0.0, 0.0, 0.0], 23),
                ([0.0, 1.0, 0.0, 0.0, 0.0], 42),
                ([0.0, 0.0, 1.0, 0.0, 0.0], 66),
                ([0.0, 1.0, 0.0, 0.0, 0.0], 42),
                ([0.0, 0.0, 1.0, 0.0, 0.0], 66),
            ],
            [
                (feature.tolist(), label.item())
                for feature, label in ds
            ]
        )
        with self.assertRaises(ValueError):
            list(ClassFeaturesDataset(ds, num_classes=2))

    def test_200_dataloader(self):
        ds = TensorDataset(torch.Tensor([
            23, 42, 66, 42, 66,
        ]).to(torch.int32))
        ds = ClassFeaturesDataset(ds, dtype=torch.float16)
        dl = DataLoader(ds, batch_size=2)
        self.assertEqual(
            [
                ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [23, 42]),
                ([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [66, 42]),
                ([[0.0, 0.0, 1.0]], [66])
            ],
            [
                (feature.tolist(), label.tolist())
                for feature, label in dl
            ]
        )

    def test_300_tuple_position(self):
        ds = TensorDataset(
            # "data"
            torch.Tensor([[1], [2], [3], [4], [5]]),
            # labels
            torch.Tensor([23, 42, 66, 42, 66]).to(torch.int32)
        )
        ds = ClassFeaturesDataset(ds, tuple_position=1)
        self.assertEqual(
            [
                ([1], [1.0, 0.0, 0.0], 23),
                ([2], [0.0, 1.0, 0.0], 42),
                ([3], [0.0, 0.0, 1.0], 66),
                ([4], [0.0, 1.0, 0.0], 42),
                ([5], [0.0, 0.0, 1.0], 66),
            ],
            [
                (data.tolist(), feature.tolist(), label.item())
                for data, feature, label in ds
            ]
        )
