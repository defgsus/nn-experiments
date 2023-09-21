import time
import unittest
import math
from itertools import permutations
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as VF

from src.datasets import *


class TestDatasetAll(unittest.TestCase):

    def test_100_iterable_single_tensor(self):
        images1_ds = TensorDataset(torch.randn(3, 1, 10, 10))
        images2_ds = TensorDataset(torch.randn(3, 1, 10, 10))
        for classes in permutations([
                IterableShuffle, TransformIterableDataset, ImagePatchIterableDataset
        ]):
            ds = InterleaveIterableDataset((images1_ds, images2_ds))

            for klass in classes:
                if klass in (IterableShuffle, ):
                    ds = klass(ds)
                elif klass is TransformIterableDataset:
                    ds = klass(ds, transforms=[lambda i: VF.pad(i, [1, 1, 1, 1])])
                elif klass is ImagePatchIterableDataset:
                    ds = klass(ds, shape=(3, 3))

            images = list(ds)
            self.assertEqual(3 * 3 * 6, len(images))
            self.assertTrue(isinstance(images[0], tuple))
            self.assertEqual(1, len(images[0]))
            # self.assertEqual((1, 3, 3), images[0][0].shape)

            dl = DataLoader(ds, batch_size=1000)
            batch = next(iter(dl))
            self.assertTrue(isinstance(batch, (tuple, list)))
            self.assertEqual(1, len(batch))

    def test_110_iterable_tensor_tuple(self):
        images1_ds = TensorDataset(torch.randn(3, 1, 10, 10), torch.randn(3, 10))
        images2_ds = TensorDataset(torch.randn(3, 1, 10, 10), torch.randn(3, 10))
        for classes in permutations([
            IterableShuffle, TransformIterableDataset, ImagePatchIterableDataset
        ]):
            ds = InterleaveIterableDataset((images1_ds, images2_ds))

            for klass in classes:
                if klass in (IterableShuffle, ):
                    ds = klass(ds)
                elif klass is TransformIterableDataset:
                    ds = klass(ds, transforms=[lambda i: VF.pad(i, [1, 1, 1, 1])])
                elif klass is ImagePatchIterableDataset:
                    ds = klass(ds, shape=(3, 3))

            num_images = 3 * 3 * 6

            images = list(ds)
            self.assertEqual(num_images, len(images))
            self.assertTrue(isinstance(images[0], tuple))
            self.assertEqual(2, len(images[0]))
            self.assertEqual((10, ), images[0][1].shape)

            dl = DataLoader(ds, batch_size=1000)
            batch = next(iter(dl))
            self.assertTrue(isinstance(batch, (tuple, list)))
            self.assertEqual(2, len(batch))
            self.assertEqual((num_images, 10), batch[1].shape)
