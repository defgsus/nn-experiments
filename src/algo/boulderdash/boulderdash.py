import io
import random
from typing import Tuple, Generator, Union, Optional

import torch
import numpy as np

from src.console import CC


class Enum:
    def __init_subclass__(cls, **kwargs):
        _key_to_value = {}
        _value_to_key = {}
        for key, value in cls.__dict__.items():
            if isinstance(value, int) and key[0].isupper():
                _key_to_value[key] = value
                if value in _value_to_key:
                    raise ValueError(f"Multiple value {value} in {cls}")
                _value_to_key[value] = key
        cls._key_to_value = _key_to_value
        cls._value_to_key = _value_to_key

    @classmethod
    def items(cls) -> Generator[Tuple[str, int], None, None]:
        yield from cls._key_to_value.items()

    @classmethod
    def count(cls) -> int:
        return len(cls._key_to_value)

    @classmethod
    def get(cls, key_or_value: Union[str, int]) -> Union[int, str]:
        if isinstance(key_or_value, str):
            return cls._key_to_value[key_or_value]
        elif isinstance(key_or_value, (int, np.int8)):
            return cls._value_to_key[key_or_value]
        else:
            raise TypeError(f"Expected str or int, got {type(key_or_value).__name__}")


class BoulderDash:

    class OBJECTS(Enum):
        Empty = 0
        Wall = 1
        Rock = 2
        Sand = 3
        Diamond = 4
        Player = 5

    class STATES(Enum):
        Nothing = 0
        Falling = 1

    class ACTIONS(Enum):
        Nop = 0
        Left = 1
        Up = 2
        Right = 3
        Down = 4

    class RESULTS(Enum):
        Nothing = 0
        Blocked = 1
        Moved = 2
        RemovedSand = 3
        PushedRock = 4
        PlayerDied = 5
        CollectedDiamond = 6
        CollectedAllDiamonds = 7

    def __init__(
            self,
            shape: Tuple[int, int],
    ):
        self.shape = shape
        self.map = np.zeros((*self.shape, 2), dtype=np.int8)

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
                obj = self.OBJECTS.get(obj)[:1]
                if obj == "E":
                    obj = "."
                if ansi_colors:
                    color = COLORS.get(obj, CC.LIGHT_GRAY)
                    if state == self.STATES.Falling:
                        obj = f"{CC.BLINK}{obj}"
                    obj = f"{color}{obj}{CC.Off}"
                print(obj, end="", file=file)
            print(file=file)

    def to_string_map(self, ansi_colors: bool = False) -> str:
        file = io.StringIO()
        self.dump(ansi_colors=ansi_colors, file=file)
        file.seek(0)
        return file.read()

    def to_image(self, tile_size: int = 8) -> np.ndarray:
        from .graphics import BoulderDashGraphics

        image = np.ndarray((3, tile_size * self.shape[0], tile_size * self.shape[1]))
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                image[:, y * tile_size: (y + 1) * tile_size, x * tile_size: (x + 1) * tile_size] \
                    = BoulderDashGraphics.graphic(int(self.map[y, x, 0]), size=tile_size)
        return image

    def to_tensor(
            self,
            one: float = 1.,
            zero: float = 0.,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Returns a tensor of shape [C, H, W] where HxW is the map size
        and C is channel planes `OBJECTS.count() + STATES.count()`.
        Each object and each state are encoded as class digits
        """
        num_objects = self.OBJECTS.count()
        num_states = self.STATES.count()

        shape = (*self.shape, num_objects + num_states)
        if zero:
            tensor = torch.empty(shape, dtype=dtype)
            torch.fill_(tensor, zero)
        else:
            tensor = torch.zeros(shape, dtype=dtype)

        map = torch.from_numpy(self.map.astype(np.int_))
        tensor.scatter_(-1, map[:, :, :1], one)
        tensor.scatter_(-1, map[:, :, 1:2] + num_objects, one)
        return tensor.permute(2, 0, 1)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "BoulderDash":
        map = tensor.permute(1, 2, 0)
        bd = BoulderDash(shape=tensor.shape[-2:])
        num_obj = bd.OBJECTS.count()
        bd_map = torch.empty(bd.map.shape, dtype=torch.int8)
        bd_map[:, :, 0] = map[:, :, :num_obj].argmax(dim=2)
        bd_map[:, :, 1] = map[:, :, num_obj:].argmax(dim=2)
        bd.map = bd_map.numpy()
        return bd

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
            raise AssertionError(f"No player on map:\n{self.to_string_map()}")

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

    def apply_physics(self) -> int:

        ret_result = self.RESULTS.Nothing

        for y in range(self.map.shape[0]):
            y = self.map.shape[0] - y - 1
            for x in range(self.map.shape[1]):
                obj = self.map[y, x, 0]
                is_falling = self.map[y, x, 1] == self.STATES.Falling

                # falling objects
                if obj in (self.OBJECTS.Rock, self.OBJECTS.Diamond):
                    has_moved = False
                    for next_y, next_x in ((y + 1, x), (y + 1, x - 1), (y + 1, x + 1)):
                        if 0 <= next_y < self.shape[0] and 0 <= next_x < self.shape[1]:
                            if self.map[next_y, next_x, 0] == self.OBJECTS.Empty:
                                if next_x == x or self.map[y, next_x, 0] == self.OBJECTS.Empty:
                                    self.map[next_y, next_x] = [obj, self.STATES.Falling]
                                    has_moved = True
                                    break

                            # rock lands directly on top of player
                            elif self.map[next_y, next_x, 0] == self.OBJECTS.Player and next_x == x and is_falling:
                                ret_result = self.RESULTS.PlayerDied
                                self.map[next_y, next_x] = [obj, self.STATES.Falling]
                                has_moved = True
                                break

                    if has_moved:
                        self.map[y, x] = [self.OBJECTS.Empty, self.STATES.Nothing]
                    else:
                        self.map[y, x, 1] = self.STATES.Nothing

        return ret_result
