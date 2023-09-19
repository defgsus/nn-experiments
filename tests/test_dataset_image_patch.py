import time
import unittest
import math
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.datasets import *


class TestDatasetImagePatch(unittest.TestCase):

    def test_100_with_tuple(self):
        image_ds = TensorDataset(torch.randn(2, 1, 10, 10))
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patches = list(patch_ds)
        self.assertEqual(3 * 3 * 2, len(patches))
        self.assertEqual((1, 3, 3), patches[0][0].shape)
        self.assertTrue(isinstance(patches[0], tuple))

    def test_110_no_tuple(self):
        image_ds = torch.randn(2, 1, 10, 10)
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patches = list(patch_ds)
        self.assertEqual(3 * 3 * 2, len(patches))
        self.assertEqual((1, 3, 3), patches[0].shape)
        self.assertTrue(isinstance(patches[0], torch.Tensor))

    def test_200_dataloader_with_tuple(self):
        image_ds = TensorDataset(torch.randn(2, 1, 10, 10))
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, (tuple, list)))
        self.assertEqual(4, patches[0].ndim)
        self.assertEqual(3 * 3 * 2, patches[0].shape[0])
        self.assertEqual((1, 3, 3), patches[0][0].shape)

    def test_200_dataloader_no_tuple(self):
        image_ds = torch.randn(2, 1, 10, 10)
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, torch.Tensor))
        self.assertEqual(4, patches.ndim)
        self.assertEqual(3 * 3 * 2, patches.shape[0])
        self.assertEqual((1, 3, 3), patches[0].shape)
