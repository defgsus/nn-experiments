import io
from typing import Tuple, Generator, List

from src.tests.base import TestBase
from src.algo.astar import astar_search


class Map:

    def __init__(self, string: str):
        self.map = [
            [0 if ch in (" ", ".") else 1 for ch in line.strip()]
            for line in string.splitlines()
            if line.strip()
        ]
        self.width = len(self.map[0])
        self.height = len(self.map)

    def astar(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        return astar_search(
            start_node=pos1,
            end_node=pos2,
            adjacent_nodes_func=self.adjacent_nodes,
            goal_cost_func=lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]),
            verbose=False,
        )

    def adjacent_nodes(self, pos: Tuple[int, int]) -> Generator[Tuple[Tuple[int, int], float], None, None]:
        for p in (
                (pos[0] - 1, pos[1]),
                (pos[0],     pos[1] - 1),
                (pos[0] + 1, pos[1]),
                (pos[0],     pos[1] + 1),
        ):
            if 0 <= p[0] < self.width and 0 <= p[1] < self.height:
                if not self.map[p[1]][p[0]]:
                    yield p, 1.

    def to_string(self):
        file = io.StringIO()
        print(" ", end="", file=file)
        for x in range(self.width):
            print(x, end="", file=file)
        print(file=file)
        for y in range(self.height):
            print(y, end="", file=file)
            for x in range(self.width):
                print("x" if self.map[y][x] else ".", end="", file=file)
            print(file=file)
        file.seek(0)
        return file.read()

class TestAStar(TestBase):

    def assert_path(self, map: str, start: Tuple[int, int], end: Tuple[int, int], path: List[Tuple[int, int]]):
        map = Map(map)
        result = map.astar(start, end)
        self.assertEqual(
            path,
            result,
            f"\nExpected:\n{path}\nGot:\n{result}\nmap:\n{map.to_string()}"
        )

    def test_100_astar(self):
        self.assert_path(
            """
            .x.x..
            .x....
            .xxx..
            ......
            """,
            (0, 0), (2, 0),
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (4, 2),
             (4, 1), (3, 1), (2, 1), (2, 0)],
        )
