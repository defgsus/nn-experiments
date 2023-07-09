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

        #print(state)

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
