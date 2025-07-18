import time
import unittest
import math
from itertools import permutations
from typing import List, Iterable

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as VF
from tqdm import tqdm

from src.datasets import *
from src.datasets.base_iterable import WrapIterableDataset


class TestDatasetAll(unittest.TestCase):

    def assert_dataset_length(self, expected_length: int, ds: Iterable):
        try:
            self.assertEqual(expected_length, len(ds))
        except TypeError:
            pass

        items_1 = list(ds)

        items_2 = []
        for item in ds:
            items_2.append(item)

        items_3 = []
        dl = DataLoader(ds, batch_size=11)
        for batch in dl:
            self.assertTrue(isinstance(batch, (tuple, list)))
            for item in zip(*batch):
                items_3.append(item)

        self.assertEqual(expected_length, len(items_1))
        self.assertEqual(expected_length, len(items_2))
        self.assertEqual(expected_length, len(items_3))

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
                    ds = klass(ds, transforms=[lambda i: VF.hflip(i)])
                elif klass is ImagePatchIterableDataset:
                    ds = klass(ds, shape=(3, 3))

            expected_num_images = 3 * 3 * 6
            self.assert_dataset_length(expected_num_images, ds)

            images_1 = list(ds)

            images_2 = []
            for item in ds:
                images_2.append(item)

            images_3 = []
            dl = DataLoader(ds, batch_size=11)
            for batch in dl:
                self.assertTrue(isinstance(batch, (tuple, list)))
                for item in zip(*batch):
                    images_3.append(item)

            for images in (images_1, images_2, images_3):
                self.assertEqual(expected_num_images, len(images), f"\nclasses: {classes}")
                self.assertTrue(isinstance(images[0], tuple))
                self.assertEqual(1, len(images[0]))
                self.assertEqual((1, 3, 3), images[0][0].shape)


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
                    ds = klass(ds, transforms=[lambda i: VF.vflip(i)])
                elif klass is ImagePatchIterableDataset:
                    ds = klass(ds, shape=(3, 3))

            expected_num_images = 3 * 3 * 6
            self.assert_dataset_length(expected_num_images, ds)

            images_1 = list(ds)

            images_2 = []
            for item in ds:
                images_2.append(item)

            images_3 = []
            dl = DataLoader(ds, batch_size=11)
            for batch in dl:
                self.assertTrue(isinstance(batch, (tuple, list)))
                for item in zip(*batch):
                    images_3.append(item)

            for images in (images_1, images_2, images_3):
                self.assertEqual(expected_num_images, len(images), f"\nclasses: {classes}")
                self.assertTrue(isinstance(images[0], tuple))
                self.assertEqual(2, len(images[0]))
                self.assertEqual((1, 3, 3), images[0][0].shape)
                self.assertEqual((10, ), images[0][1].shape)

    def test_repeat(self):
        ds = TensorDataset(torch.randn(3, 3, 5, 5), torch.randn(3, 10))
        ds = WrapDataset(ds)
        self.assert_dataset_length(3, ds)
        self.assert_dataset_length(3, ds.repeat(1))
        self.assert_dataset_length(6, ds.repeat(2))
        self.assert_dataset_length(30, ds.repeat(10))

    def test_repeat_iterable(self):
        ds = TensorDataset(torch.randn(3, 3, 5, 5), torch.randn(3, 10))
        ds = WrapIterableDataset(ds)
        self.assert_dataset_length(3, ds)
        self.assert_dataset_length(3, ds.repeat(1))
        self.assert_dataset_length(6, ds.repeat(2))
        self.assert_dataset_length(30, ds.repeat(10))
