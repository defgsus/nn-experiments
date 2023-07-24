import time
import unittest
import math
from typing import List

import torch
from torch.utils.data import Dataset, IterableDataset

from src.datasets import InterleaveIterableDataset


class NumbersDataset(Dataset):

    def __init__(self, start: int = 0, end: int = 10):
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, item):
        return item


class LettersIterableDataset(IterableDataset):

    def __init__(self, start: str = 'a', end: str = 'f'):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(ord(self.start), ord(self.end) + 1):
            yield chr(i)


class TestInterleaveDataset(unittest.TestCase):

    def test_100_interleave_iterable(self):
        self.assertEqual(
            [0, "a", 1, "b", 2, "c", 3, "d", 4, "e", 5, "f"],
            list(InterleaveIterableDataset(
                datasets=[
                    NumbersDataset(0, 5),
                    LettersIterableDataset('a', 'f'),
                ]
            ))
        )
        self.assertEqual(
            [0, "a", 1, "b", 2, "c", 3, "d", "e", "f"],
            list(InterleaveIterableDataset(
                datasets=[
                    NumbersDataset(0, 3),
                    LettersIterableDataset('a', 'f'),
                ]
            ))
        )

    def test_110_interleave_iterable_shuffle(self):
        unshuffled = [0, "a", 1, "b", 2, "c", 3, "d", "e", "f"]
        shuffled = list(InterleaveIterableDataset(
            datasets=[
                NumbersDataset(0, 3),
                LettersIterableDataset('a', 'f'),
            ],
            shuffle_datasets=True,
        ))
        self.assertNotEqual(
            unshuffled,
            shuffled,
        )
        self.assertEqual(len(unshuffled), len(shuffled))
        for i in unshuffled:
            self.assertIn(i, shuffled)

    def test_120_interleave_iterable_counts(self):
        self.assertEqual(
            [0, "a", "b", 1, "c", "d", 2, "e", "f", 3, 4, 5],
            list(InterleaveIterableDataset(
                datasets=[
                    NumbersDataset(0, 5),
                    LettersIterableDataset('a', 'f'),
                ],
                counts=[1, 2],
            ))
        )
        self.assertEqual(
            [0, 1, 2, "a", "b", 3, 4, 5, "c", "d", "e", "f"],
            list(InterleaveIterableDataset(
                datasets=[
                    NumbersDataset(0, 5),
                    LettersIterableDataset('a', 'f'),
                ],
                counts=[3, 2],
            ))
        )
