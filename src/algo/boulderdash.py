import io
import random
from typing import Tuple

import numpy as np

from src.console import CC


class BoulderDash:

    class OBJECTS:
        Empty = 0
        Wall = 1
        Rock = 2
        Sand = 3
        Diamond = 4
        Player = 5

    class STATES:
        Nothing = 0
        Falling = 1

    class ACTIONS:
        Nop = 0
        Left = 1
        Up = 2
        Right = 3
        Down = 4

    class RESULTS:
        Nothing = 0
        Blocked = 1
        Moved = 2
        RemovedSand = 3
        PushedRock = 4
        PlayerDied = 5
        CollectedDiamond = 6
        CollectedAllDiamonds = 7

    NUMBER_TO_OBJECT = {
        value: key
        for key, value in OBJECTS.__dict__.items()
        if isinstance(value, int)
    }

    def __init__(
            self,
            shape: Tuple[int, int],
    ):
        self.shape = shape
        self.map = np.zeros((*self.shape, 2), dtype=np.dtype)

    @classmethod
    def from_string_map(cls, string: str) -> "BoulderDash":
        lines = [l.strip() for l in string.splitlines()]
        lines = list(filter(bool, lines))
        assert lines, f"No data in string map"
        height = len(lines)
        width = max(len(l) for l in lines)

        bd = cls(shape=(height, width))
        for y in range(height):
            for x in range(width):
                ch = lines[y][x].upper() if x < len(lines[y]) else "E"
                obj = cls.OBJECTS.Empty
                for key, num in cls.OBJECTS.__dict__.items():
                    if key.startswith(ch):
                        obj = num
                        break
                bd.map[y, x, 0] = obj

        return bd

    @classmethod
    def from_random(
            cls,
            shape: Tuple[int, int],
            ratio_wall: float = .15,
            ratio_rock: float = 0.05,
            ratio_diamond: float = .01,
            ratio_sand: float = 0.2,
            with_border: bool = False,
    ) -> "BoulderDash":
        bd = cls(shape=shape)
        rng = random.Random()
        area = bd.shape[0] * bd.shape[1]
        if with_border:
            assert bd.shape[0] > 2 and bd.shape[1] > 2, f"Got: {bd.shape}"
            area = (bd.shape[0] - 2) * (bd.shape[1] - 2)
            for y in range(bd.shape[0]):
                bd.map[y, 0, 0] = bd.OBJECTS.Wall
                bd.map[y, bd.shape[1] - 1, 0] = bd.OBJECTS.Wall
            for x in range(bd.shape[1]):
                bd.map[0, x, 0] = bd.OBJECTS.Wall
                bd.map[bd.shape[0] - 1, x, 0] = bd.OBJECTS.Wall

        def _place_random(obj: int):
            for i in range(area * 100):
                y, x = rng.randint(0, bd.shape[0] - 1), rng.randint(0, bd.shape[1] - 1)
                if bd.map[y, x, 0] == bd.OBJECTS.Empty:
                    bd.map[y, x, 0] = obj
                    break

        _place_random(bd.OBJECTS.Player)

        for ratio, obj in (
                (ratio_wall, bd.OBJECTS.Wall),
                (ratio_rock, bd.OBJECTS.Rock),
                (ratio_diamond, bd.OBJECTS.Diamond),
                (ratio_sand, bd.OBJECTS.Sand),
        ):
            if ratio > 0:
                for i in range(max(1, int(ratio * bd.shape[0] * bd.shape[1]))):
                    _place_random(obj)

        return bd

    def dump(self, file=None, ansi_colors: bool = False):
        COLORS = {
            "W": CC.LIGHT_GRAY,
            "R": CC.BROWN,
            "S": CC.YELLOW,
            "D": CC.LIGHT_CYAN,
            "P": CC.GREEN,
        }
        for row in self.map:
            for obj, state in row:
                obj = self.NUMBER_TO_OBJECT[obj][:1]
                if obj == "E":
                    obj = "."
                if ansi_colors:
                    color = COLORS.get(obj, CC.LIGHT_GRAY)
                    obj = f"{obj}{state}"
                    if state == self.STATES.Falling:
                        obj = f"{CC.UNDERLINE}{obj}"
                    obj = f"{color}{obj}{CC.Off}"
                print(obj, end="", file=file)
            print(file=file)

    def to_string(self, ansi_colors: bool = False) -> str:
        file = io.StringIO()
        self.dump(ansi_colors=ansi_colors, file=file)
        file.seek(0)
        return file.read()

    def player_position(self) -> Tuple[int, int]:
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x, 0] == self.OBJECTS.Player:
                    return y, x

    def num_diamonds(self):
        return np.sum(self.map[:, :, 0] == self.OBJECTS.Diamond)

    def apply_action(self, action: int = ACTIONS.Nop) -> int:
        if action == self.ACTIONS.Nop:
            return self.RESULTS.Nothing

        pos = self.player_position()
        if pos is None:
            raise AssertionError(f"No player on map:\n{self.to_string()}")

        if action in (self.ACTIONS.Up, self.ACTIONS.Down):
            ofs = -1 if action == self.ACTIONS.Up else 1
            if (ofs == -1 and pos[0] < 1) or (ofs == 1 and pos[0] >= self.shape[0] - 1):
                return self.RESULTS.Blocked

            obj = self.map[pos[0] + ofs, pos[1], 0]
            if obj in (self.OBJECTS.Empty, self.OBJECTS.Sand, self.OBJECTS.Diamond):
                is_diamond = obj == self.OBJECTS.Diamond
                self.map[pos[0] + ofs, pos[1]] = [self.OBJECTS.Player, self.STATES.Nothing]
                self.map[pos[0], pos[1]] = [self.OBJECTS.Empty, self.STATES.Nothing]
                if is_diamond:
                    if self.num_diamonds() == 0:
                        return self.RESULTS.CollectedAllDiamonds
                    else:
                        return self.RESULTS.CollectedDiamond
                return self.RESULTS.RemovedSand if obj == self.OBJECTS.Sand else self.RESULTS.Moved
            else:
                return self.RESULTS.Blocked

        if action in (self.ACTIONS.Left, self.ACTIONS.Right):
            ofs = -1 if action == self.ACTIONS.Left else 1
            if (ofs == -1 and pos[1] < 1) or (ofs == 1 and pos[1] >= self.shape[1] - 1):
                return self.RESULTS.Blocked

            obj = self.map[pos[0], pos[1] + ofs, 0]
            if obj in (self.OBJECTS.Empty, self.OBJECTS.Sand, self.OBJECTS.Diamond):
                is_diamond = obj == self.OBJECTS.Diamond
                self.map[pos[0], pos[1] + ofs] = [self.OBJECTS.Player, self.STATES.Nothing]
                self.map[pos[0], pos[1]] = [self.OBJECTS.Empty, self.STATES.Nothing]
                if is_diamond:
                    if self.num_diamonds() == 0:
                        return self.RESULTS.CollectedAllDiamonds
                    else:
                        return self.RESULTS.CollectedDiamond
                return self.RESULTS.RemovedSand if obj == self.OBJECTS.Sand else self.RESULTS.Moved

            elif obj == self.OBJECTS.Rock:
                x = pos[1] + ofs
                moveable = False
                while 0 < x < self.shape[1] - 1:
                    x += ofs
                    if self.map[pos[0], x, 0] == self.OBJECTS.Rock:
                        pass
                    elif self.map[pos[0], x, 0] == self.OBJECTS.Empty:
                        moveable = True
                        break
                    else:
                        break
                if not moveable:
                    return self.RESULTS.Blocked

                self.map[pos[0], x] = [self.OBJECTS.Rock, self.STATES.Nothing]
                self.map[pos[0], pos[1] + ofs] = [self.OBJECTS.Player, self.STATES.Nothing]
                self.map[pos[0], pos[1]] = [self.OBJECTS.Empty, self.STATES.Nothing]
                return self.RESULTS.PushedRock

    def step(self) -> int:

        # copy wall, sand & player already to next map
        next_map = np.zeros_like(self.map)
        do_copy = (
            (self.map[:, :, 0] == self.OBJECTS.Wall)
            | (self.map[:, :, 0] == self.OBJECTS.Sand)
            | (self.map[:, :, 0] == self.OBJECTS.Player)
        )
        next_map[do_copy, 0] = self.map[do_copy, 0]

        ret_result = self.RESULTS.Nothing

        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                obj = self.map[y, x, 0]
                is_falling = self.map[y, x, 1] == self.STATES.Falling

                # falling objects
                if obj in (self.OBJECTS.Rock, self.OBJECTS.Diamond):
                    has_moved = False
                    for next_y, next_x in ((y + 1, x), (y + 1, x - 1), (y + 1, x + 1)):
                        if 0 <= next_y < self.shape[0] and 0 <= next_x < self.shape[1]:
                            if self.map[next_y, next_x, 0] == self.OBJECTS.Empty and next_map[next_y, next_x, 0] == self.OBJECTS.Empty:
                                if next_x == x or (self.map[y, next_x, 0] == self.OBJECTS.Empty and next_map[y, next_x, 0] == self.OBJECTS.Empty):
                                    next_map[next_y, next_x] = [obj, self.STATES.Falling]
                                    has_moved = True
                                    break

                            # rock lands directly on top of player
                            elif next_map[next_y, next_x, 0] == self.OBJECTS.Player and next_x == x and is_falling:
                                ret_result = self.RESULTS.PlayerDied
                                next_map[next_y, next_x] = [obj, self.STATES.Falling]
                                has_moved = True
                                break

                    if not has_moved:
                        next_map[y, x, 0] = obj

        self.map = next_map
        return ret_result


if __name__ == "__main__":
    def run_cli():
        bd = BoulderDash.from_random((16, 16))
        while True:
            print()
            bd.dump(ansi_colors=True)
            cmd = input("\nw/a/s/d> ").lower()
            action = bd.ACTIONS.Nop
            if cmd == "q":
                break
            elif cmd == "r":
                bd = bd.from_random(bd.shape)
                continue
            elif cmd == "w":
                action = bd.ACTIONS.Up
            elif cmd == "a":
                action = bd.ACTIONS.Left
            elif cmd == "s":
                action = bd.ACTIONS.Down
            elif cmd == "d":
                action = bd.ACTIONS.Right

            result1 = bd.apply_action(action)
            result2 = bd.step()

            for key, value in bd.RESULTS.__dict__.items():
                if value == result1:
                    r1 = key
                if value == result2:
                    r2 = key
            print(f"result: {r1}, {r2}")
            if result2 == bd.RESULTS.PlayerDied:
                print("HIT AND DIED!!")

    run_cli()
