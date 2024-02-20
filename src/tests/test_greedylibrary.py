import io
import random
import unittest

import torch

from src.algo.greedylibrary import *
from src.tests.base import TestBase


class TestGreedyLibrary(TestBase):

    def test_100_copy(self):
        l1 = GreedyLibrary(3, (4, 5))
        l2 = l1.copy()

        self.assertNotEqual(l1, l2)
        self.assertEqual(l1.entries.tolist(), l2.entries.tolist())
        self.assertNotEqual(id(l1.entries), id(l2.entries))

        l1.entries[0][0][0] = 23.
        self.assertNotEqual(l1.entries.tolist(), l2.entries.tolist())

    def test_110_save_torch(self):
        for ndim in range(1, 5):
            shape = tuple(random.randint(3, 7) for i in range(ndim))
            lib = GreedyLibrary(random.randint(10, 100), shape)
            lib.fit(torch.randn(500, *shape))

            entries = lib.entries.tolist()
            hits = lib.hits

            f = io.BytesIO()
            lib.save_torch(f)

            f.seek(0)
            lib = GreedyLibrary.from_torch(f)

            self.assertEqual(entries, lib.entries.tolist())
            self.assertEqual(hits, lib.hits)
            self.assertEqual(shape, lib.shape)

    def test_120_sort(self):
        cl = GreedyLibrary(5, (1,))
        cl.entries = torch.Tensor([[1], [2], [3], [4], [5]])
        cl.hits = [4, 0, 3, 1, 2]

        self.assertEqual([0, 1, 2, 3, 4], cl.sort_entries().hits)
        self.assertEqual([[2], [4], [5], [3], [1]], cl.sort_entries().entries.tolist())
        self.assertEqual([4, 3, 2, 1, 0], cl.sort_entries(reverse=True).hits)
        self.assertEqual([[1], [3], [5], [4], [2]], cl.sort_entries(reverse=True).entries.tolist())

        # original stays unchanged
        self.assertEqual([4, 0, 3, 1, 2], cl.hits)
        self.assertEqual([[1], [2], [3], [4], [5]], cl.entries.tolist())

    def test_200_match(self):
        cl = GreedyLibrary(5, (1,))
        cl.entries = torch.Tensor([
            [1], [2], [3], [4], [5]
        ])
        self.assertEqual(
            [0, 1, 2, 3, 4],
            list(cl.best_entries_for(cl.entries)[0])
        )
        self.assertEqual(
            [0, 4, 1, 2],
            list(cl.best_entries_for(torch.Tensor([[-10], [10], [2.49], [2.51]]))[0])
        )

    def test_210_match_skip(self):
        cl = GreedyLibrary(5, (1,))
        cl.entries = torch.Tensor([
            [1], [2], [3], [4], [5]
        ])
        cl.fit(torch.Tensor([[1], [1], [2]]))
        self.assertEqual([2, 1, 0, 0, 0], cl.hits)

        self.assertEqual(
            [1, 1, 2, 3, 4],
            list(cl.best_entries_for(cl.entries, skip_top_entries=True)[0])
        )
        self.assertEqual(
            [1, 1, 2, 3, 4],
            list(cl.best_entries_for(cl.entries, skip_top_entries=1)[0])
        )
        self.assertEqual(
            [2, 2, 2, 3, 4],
            list(cl.best_entries_for(cl.entries, skip_top_entries=2)[0])
        )

    def test_300_ndim(self):
        for shape in (
                (1,),
                (10,),
                (3, 10),
                (5, 9, 20),
                (3, 5, 9, 20),
        ):
            cl = GreedyLibrary(100, shape)
            cl.fit(torch.randn(1000, *shape))
            cl.fit(torch.randn(1000, *shape), zero_mean=True)
            if 1 < len(shape) < 4:
                cl.convolve(torch.randn(1, *shape))

    @unittest.skipIf(not torch.cuda.is_available(), "no cuda available")
    def test_400_cuda(self):
        lib = GreedyLibrary(50, (3, 2), device="cuda")

        # add all things here that can be done with GreedyLibrary
        lib.fit(torch.randn(100, *lib.shape))
        lib.fit(torch.randn(100, *lib.shape), zero_mean=True)
        lib.fit(torch.randn(100, *lib.shape), skip_top_entries=3)
        lib.fit(torch.randn(100, *lib.shape), grow_if_distance_above=0)
        lib.fit(torch.randn(100, *lib.shape), grow_if_distance_above=0, skip_top_entries=5)
        lib.convolve(torch.randn(4, 3, 2))
        lib.sorted_entry_indices("hits")
        lib.sorted_entry_indices("tsne")
        lib.plot_entries()
        lib.drop_entries(hits_lt=20, inplace=True)

    @unittest.skipIf(not torch.cuda.is_available(), "no cuda available")
    def test_410_cuda_copy(self):
        def _assert_device(lib, device):
            device = to_torch_device(device)
            self.assertEqual(device, lib.device)
            self.assertEqual(device.type, lib.entries.device.type)

        lib = GreedyLibrary(50, (3, 2), device="cuda")
        self.assertEqual("cuda", lib.device.type)
        expected_device = lib.device
        _assert_device(lib, expected_device)
        _assert_device(lib.copy(), expected_device)
        _assert_device(lib.sort_entries(), expected_device)
        _assert_device(lib.drop_unused(), expected_device)

        _assert_device(lib.to("cpu"), "cpu")
        _assert_device(lib, expected_device)

    def test_500_convolve(self):
        #cl = GreedyLibrary(5, (3,))
        #self.assertEqual((2, 8), cl.convolve(torch.ones(2, 10)).shape)

        cl = GreedyLibrary(5, (1, 3, 4))
        self.assertEqual((5, 8, 7), cl.convolve(torch.ones(1, 10, 10)).shape)
        cl = GreedyLibrary(5, (3, 3, 4))
        self.assertEqual((5, 8, 7), cl.convolve(torch.ones(3, 10, 10)).shape)

