from .wangtiles import WangTiles


def get_wang_tile_colors(num_colors: int, num_places: int = 4):
    def _shift(idx, i):
        for _ in range(i):
            idx //= num_colors
        return idx
    ret = []
    for idx in range(num_colors ** num_places):
        ret.append(
            [_shift(idx, i) % num_colors for i in range(num_places)]
        )
    return ret


class WangTiles2E(WangTiles):
    """All wang tiles with 2 colors (0 or 1) on edges"""
    def __init__(self):
        super().__init__(
            colors=get_wang_tile_colors(2),
            mode="edge",
        )


class WangTiles2C(WangTiles):
    """All wang tiles with 2 colors (0 or 1) on corners"""
    def __init__(self):
        super().__init__(
            colors=get_wang_tile_colors(2),
            mode="corner",
        )


class WangTiles3E(WangTiles):
    """All wang tiles with 3 colors (0, 1 or 2) on edges"""
    def __init__(self):
        super().__init__(
            colors=get_wang_tile_colors(3),
            mode="edge",
        )
