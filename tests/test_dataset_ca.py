import time
import unittest
import math
from typing import List

import torch

from src.datasets import TotalCADataset


class TestDatasetCA(unittest.TestCase):

    def test_100_init_prob(self):
        shape = (1024, 1024)
        for dtype in (torch.uint8, torch.float32):
            for init_prob in (0., .1, .3, .5, .7, .9, 1.):
                ds = TotalCADataset(
                    shape,
                    init_prob=init_prob,
                    dtype=dtype,
                )
                cells = ds.init_cells()
                num_alive = int(cells.sum())
                ratio = num_alive / math.prod(shape)
                self.assertAlmostEqual(init_prob, ratio, places=2)

    def test_200_total(self):
        shape = (32, 32)
        ds = TotalCADataset(shape, init_prob=.7)
        state, rule = ds["3-23"]

        self.assertEqual(
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [int(r) for r in rule]
        )

    def test_210_total_repeat(self):
        shape = (32, 32)
        ds = TotalCADataset(
            shape,
            init_prob=(0, 1),
            num_iterations=(2, 10),
            seed=23,
        )
        state, rule = ds["3-23"]

        self.assertEqual(
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [int(r) for r in rule]
        )

    def assert_ca(
            self,
            input_and_output: str,
            rule: str = "3-23",
            wrap: bool = False,
            dtype: torch.dtype = torch.uint8,
    ):
        def _convert(lines: List[str], index: int) -> torch.Tensor:
            return torch.Tensor([
                [1 if c == "#" else 0 for c in line.split("|")[index]]
                for line in lines
            ]).to(dtype)

        lines = [
            line.strip() for line in input_and_output.splitlines()
            if line.strip()
        ]

        input = _convert(lines, 0)
        expected_outputs = [
            _convert(lines, i + 1)
            for i in range(0, lines[0].count("|"))
        ]

        ds = TotalCADataset(
            input.shape,
            wrap=wrap,
            num_iterations=len(expected_outputs),
            dtype=dtype,
        )
        birth, survive = ds.index_to_rule(ds.rule_to_index(rule))

        state = input
        for i, expected_output in enumerate(expected_outputs):
            state = ds.step_cells(state, birth, survive)
            self.assertEqual(
                expected_output.tolist(),
                state.tolist(),
                f"\nExpected (in iteration {i}):\n{expected_output}\nGot:\n{state}"
            )

    def test_300_conways_game_of_life(self):
        for dtype in (torch.uint8, torch.float):
            self.assert_ca("""
                .....|.....|..#..
                ..#..|.###.|.#.#.
                .###.|.#.#.|#...#
                ..#..|.###.|.#.#.
                .....|.....|..#..
            """, dtype=dtype)
            self.assert_ca("""
                .#...|.....|.....|.....|.....|.....|.....|.....|.....|.....|.....|.....|.....
                ..#..|#.#..|..#..|.#...|..#..|.....|.....|.....|.....|.....|.....|.....|.....
                ###..|.##..|#.#..|..##.|...#.|.#.#.|...#.|..#..|...#.|.....|.....|.....|.....
                .....|.#...|.##..|.##..|.###.|..##.|.#.#.|...##|....#|..#.#|....#|...##|...##
                .....|.....|.....|.....|.....|..#..|..##.|..##.|..###|...##|...##|...##|...##
            """, dtype=dtype)

            self.assert_ca("""
                .#...|.....|.....|.....|.....|.....|.....|.....|.....|...#.|...##|...##|#..##|#...#|#..#.|##...|.#...
                ..#..|#.#..|..#..|.#...|..#..|.....|.....|.....|.....|.....|.....|.....|.....|....#|#...#|#...#|##..#
                ###..|.##..|#.#..|..##.|...#.|.#.#.|...#.|..#..|...#.|.....|.....|.....|.....|.....|.....|.....|.....
                .....|.#...|.##..|.##..|.###.|..##.|.#.#.|...##|....#|..#.#|....#|...#.|....#|.....|.....|.....|.....
                .....|.....|.....|.....|.....|..#..|..##.|..##.|..###|...##|..#.#|#...#|#....|#..#.|#....|....#|#....
            """, wrap=True, dtype=dtype)

    def test_400_speed(self):
        """
        first implementation:
            (32, 32)     torch.uint8      wrap=False cells/s=13,902,945
            (32, 32)     torch.uint8      wrap=True  cells/s=9,108,662
            (32, 32)     torch.float32    wrap=False cells/s=13,455,411
            (32, 32)     torch.float32    wrap=True  cells/s=8,115,368
            (128, 128)   torch.uint8      wrap=False cells/s=100,147,677
            (128, 128)   torch.uint8      wrap=True  cells/s=74,320,733
            (128, 128)   torch.float32    wrap=False cells/s=87,735,018
            (128, 128)   torch.float32    wrap=True  cells/s=61,928,547
            (1024, 1024) torch.uint8      wrap=False cells/s=412,475,663
            (1024, 1024) torch.uint8      wrap=True  cells/s=350,573,802
            (1024, 1024) torch.float32    wrap=False cells/s=113,898,721
            (1024, 1024) torch.float32    wrap=True  cells/s=95,649,338
        """
        print()
        for shape in (
                (32, 32),
                (128, 128),
                (1024, 1024),
        ):
            for dtype in (torch.uint8, torch.float):
                for wrap in (False, True):
                    ds = TotalCADataset(shape=shape, wrap=wrap, dtype=dtype)

                    cells = ds.init_cells()
                    iterations = 0

                    start_time = time.time()
                    while True:
                        for i in range(100):
                            cells = ds.step_cells(cells, [3], [2, 3])
                            iterations += 1

                        cur_time = time.time()
                        if cur_time - start_time > .1:
                            break

                    seconds = cur_time - start_time
                    cells = math.prod(shape) * iterations
                    cells_per_second = int(cells / seconds)

                    print(f"{str(shape):12} {str(dtype):16} wrap={str(wrap):5} cells/s={cells_per_second:,}")
