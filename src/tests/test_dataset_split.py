import torch
from tqdm import tqdm

from torch.utils.data import TensorDataset, IterableDataset

from tests.base import *
from src.datasets import SplitIterableDataset


class NumberIterableDataset(IterableDataset):

    def __init__(self, count):
        self.count = count

    def __iter__(self):
        yield from range(self.count)


class TestDatasetSplit(TestBase):

    def test_100_split_iterable(self):
        ds = NumberIterableDataset(10)

        ds_train = SplitIterableDataset(ds, ratio=3, train=True)
        ds_test = SplitIterableDataset(ds, ratio=3, train=False)

        self.assertEqual([0, 1, 2, 4, 5, 6, 8, 9], list(ds_train))
        self.assertEqual([3, 7], list(ds_test))

    def test_200_split_dataset(self):
        ds = TensorDataset(torch.linspace(0, 9, 10, dtype=torch.int64))

        ds_train = SplitIterableDataset(ds, ratio=3, train=True)
        ds_test = SplitIterableDataset(ds, ratio=3, train=False)

        self.assertEqual([0, 1, 2, 4, 5, 6, 8, 9], [int(i[0]) for i in ds_train])
        self.assertEqual([3, 7], [int(i[0]) for i in ds_test])
