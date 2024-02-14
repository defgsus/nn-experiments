import time
import unittest
import math
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.datasets import *
from .base import TestBase


class TestDatasetImagePatch(TestBase):

    def test_100_with_tuple(self):
        image_ds = TensorDataset(torch.randn(2, 1, 10, 10))  # TensorDataset always yields tuples
        self.assertIsInstance(next(iter(image_ds)), tuple)
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patches = list(patch_ds)
        self.assertEqual(3 * 3 * 2, len(patches))
        self.assertEqual((1, 3, 3), patches[0][0].shape)
        self.assertTrue(isinstance(patches[0], tuple))

    def test_101_with_tuple_and_pos(self):
        image_ds = TensorDataset(torch.randn(2, 1, 10, 10))
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3), with_pos=True)
        patches = list(patch_ds)
        self.assertEqual(3 * 3 * 2, len(patches))
        self.assertEqual((1, 3, 3), patches[0][0].shape)
        self.assertTrue(isinstance(patches[0], tuple))
        self.assertTensorEqual([0, 0], patches[0][1])
        self.assertTrue(isinstance(patches[0][1], torch.Tensor))
        self.assertEqual(torch.int64, patches[0][1].dtype)

    def test_110_no_tuple(self):
        image_ds = torch.randn(2, 1, 10, 10)
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patches = list(patch_ds)
        self.assertEqual(3 * 3 * 2, len(patches))
        self.assertEqual((1, 3, 3), patches[0].shape)
        self.assertTrue(isinstance(patches[0], torch.Tensor))

    def test_111_no_tuple_and_pos(self):
        image_ds = torch.randn(2, 1, 10, 10)
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3), with_pos=True)
        patches = list(patch_ds)
        self.assertEqual(3 * 3 * 2, len(patches))
        self.assertEqual((1, 3, 3), patches[0][0].shape)
        self.assertTrue(isinstance(patches[0], tuple))
        self.assertTensorEqual([0, 0], patches[0][1])
        self.assertTrue(isinstance(patches[0][1], torch.Tensor))
        self.assertEqual(torch.int64, patches[0][1].dtype)

    def test_200_dataloader_with_tuple(self):
        image_ds = TensorDataset(torch.randn(2, 1, 10, 10))
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, (tuple, list)))
        self.assertEqual(4, patches[0].ndim)
        self.assertEqual(3 * 3 * 2, patches[0].shape[0])
        self.assertEqual((1, 3, 3), patches[0][0].shape)

    def test_201_dataloader_with_tuple_and_pos(self):
        image_ds = TensorDataset(torch.randn(2, 1, 10, 10))
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3), with_pos=True)
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, (tuple, list)))
        self.assertEqual(4, patches[0].ndim)
        self.assertEqual(3 * 3 * 2, patches[0].shape[0])
        self.assertEqual((1, 3, 3), patches[0][0].shape)
        self.assertTensorEqual(
            [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6],
             [0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]],
            patches[1],
        )

    def test_210_dataloader_no_tuple(self):
        image_ds = torch.randn(2, 1, 10, 10)
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(3, 3))
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, torch.Tensor))
        self.assertEqual(4, patches.ndim)
        self.assertEqual(3 * 3 * 2, patches.shape[0])
        self.assertEqual((1, 3, 3), patches[0].shape)

    def test_300_interleave_images(self):
        image_ds = [
            [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]],
            [[[2.1, 2.2, 2.3], [2.4, 2.5, 2.6], [2.7, 2.8, 2.9]]],
            [[[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]]],
        ]
        image_ds = [torch.Tensor(i) for i in image_ds]

        # -- no interleaving --
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(2, 2), stride=1)
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, torch.Tensor))
        self.assertEqual(4, patches.ndim)
        self.assertTensorEqual(
            [
                [[[1.1000, 1.2000], [1.4000, 1.5000]]],
                [[[1.2000, 1.3000], [1.5000, 1.6000]]],

                [[[2.1000, 2.2000], [2.4000, 2.5000]]],
                [[[2.2000, 2.3000], [2.5000, 2.6000]]],
                [[[2.4000, 2.5000], [2.7000, 2.8000]]],
                [[[2.5000, 2.6000], [2.8000, 2.9000]]],

                [[[3.1000, 3.2000], [3.4000, 3.5000]]],
                [[[3.2000, 3.3000], [3.5000, 3.6000]]],
            ],
            patches,
        )

        # -- interleave 2 --
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(2, 2), stride=1, interleave_images=2)
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, torch.Tensor))
        self.assertEqual(4, patches.ndim)
        self.assertTensorEqual(
            [
                [[[1.1000, 1.2000], [1.4000, 1.5000]]],
                [[[2.1000, 2.2000], [2.4000, 2.5000]]],
                [[[1.2000, 1.3000], [1.5000, 1.6000]]],
                [[[2.2000, 2.3000], [2.5000, 2.6000]]],
                [[[3.1000, 3.2000], [3.4000, 3.5000]]],
                [[[2.4000, 2.5000], [2.7000, 2.8000]]],
                [[[3.2000, 3.3000], [3.5000, 3.6000]]],
                [[[2.5000, 2.6000], [2.8000, 2.9000]]],
            ],
            patches,
        )

        # -- interleave 3 --
        patch_ds = ImagePatchIterableDataset(image_ds, shape=(2, 2), stride=1, interleave_images=3)
        patch_dl = DataLoader(patch_ds, batch_size=1000)
        patches = next(iter(patch_dl))
        self.assertTrue(isinstance(patches, torch.Tensor))
        self.assertEqual(4, patches.ndim)
        self.assertTensorEqual(
            [
                [[[1.1000, 1.2000], [1.4000, 1.5000]]],
                [[[2.1000, 2.2000], [2.4000, 2.5000]]],
                [[[3.1000, 3.2000], [3.4000, 3.5000]]],

                [[[1.2000, 1.3000], [1.5000, 1.6000]]],
                [[[2.2000, 2.3000], [2.5000, 2.6000]]],
                [[[3.2000, 3.3000], [3.5000, 3.6000]]],

                [[[2.4000, 2.5000], [2.7000, 2.8000]]],
                [[[2.5000, 2.6000], [2.8000, 2.9000]]],
            ],
            patches,
        )