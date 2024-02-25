from typing import Iterable, Tuple, Optional, Union

import torch


class WangTemplate:
    def __init__(
            self,
            indices: Iterable[Iterable[int]],
            image: torch.Tensor,
    ):
        self.indices = torch.Tensor(indices).to(torch.int64)
        self.image = image
        self._index_to_pos = {}
        for y, row in enumerate(self.indices):
            for x, idx in enumerate(row):
                self._index_to_pos[int(idx)] = (int(y), int(x))

    def __repr__(self):
        return f"WangTemplate(shape={tuple(self.shape)}, tile_shape={self.tile_shape})"

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape

    @property
    def tile_shape(self) -> Tuple[int, int, int]:
        return (
            self.image.shape[-3],
            self.image.shape[-2] // self.indices.shape[-2],
            self.image.shape[-1] // self.indices.shape[-1],
        )

    def tile(
            self,
            index: Optional[int] = None,
            position: Optional[Tuple[int, int]] = None,
    ):
        if index is None and position is None:
            raise ValueError(f"Expected `index` or `position`, got none")
        if index is not None and position is not None:
            raise ValueError(f"Expected `index` or `position`, got both")

        if position is not None:
            pos = position
        else:
            pos = self._index_to_pos[int(index)]

        shape = self.tile_shape
        return self.image[
           ...,
           pos[0] * shape[-2]: (pos[0] + 1) * shape[-2],
           pos[1] * shape[-1]: (pos[1] + 1) * shape[-1],
        ]

    def render_map(
            self,
            tile_indices: torch.Tensor,
            overlap: Union[int, Tuple[int, int]] = 0,
    ):
        from .render import render_wang_map
        return render_wang_map(
            wang_template=self,
            tile_indices=tile_indices,
            overlap=overlap,
        )


OPTIMAL_WANG_INDICES_SQUARE = {
    ("edge", 2): [
        [4, 6, 14, 12],
        [5, 7, 15, 13],
        [1, 3, 11, 9],
        [0, 2, 10, 8],
    ],
    ("corner", 2): [
        [4,  3, 14, 6],
        [10, 7, 15, 13],
        [1,  9, 11, 12],
        [0,  2, 5,  8],
    ],
    ("edge", 3): [
        [18, 21, 48, 45, 24, 75, 51, 78, 72],
        [20, 23, 50, 47, 26, 77, 53, 80, 74],
        [11, 14, 41, 38, 17, 68, 44, 71, 65],
        [19, 22, 49, 46, 25, 76, 52, 79, 73],
        [ 2,  5, 32, 29,  8, 59, 35, 62, 56],
        [ 9, 12, 39, 36, 15, 66, 42, 69, 63],
        [10, 13, 40, 37, 16, 67, 43, 70, 64],
        [ 1,  4, 31, 28,  7, 58, 34, 61, 55],
        [ 0,  3, 30, 27,  6, 57, 33, 60, 54],
    ],
    ("corner", 3): [
        [58, 43, 47, 60, 24, 20, 61, 49, 38],
        [10, 35, 75, 11, 56, 57, 17, 73, 30],
        [45,  5, 70, 48, 12, 16, 50, 63,  7],
        [78, 22, 44, 79, 52, 53, 76, 42, 26],
        [65, 55, 32, 71, 80, 77, 64, 29, 59],
        [39,  9,  4, 41, 68, 67, 36,  3, 13],
        [34, 51, 19, 31, 40, 37, 33, 25, 46],
        [ 8, 74, 54,  1, 28, 27,  2, 62, 72],
        [23, 66, 15, 18,  0,  6, 21, 14, 69],
    ],
}


class WangTemplate2E(WangTemplate):
    def __init__(
            self,
            image: torch.Tensor,
    ):
        super().__init__(
            indices=OPTIMAL_WANG_INDICES_SQUARE[("edge", 2)],
            image=image,
        )


class WangTemplate2C(WangTemplate):
    def __init__(
            self,
            image: torch.Tensor,
    ):
        super().__init__(
            indices=OPTIMAL_WANG_INDICES_SQUARE[("corner", 2)],
            image=image,
        )


class WangTemplate3E(WangTemplate):
    def __init__(
            self,
            image: torch.Tensor,
    ):
        super().__init__(
            indices=OPTIMAL_WANG_INDICES_SQUARE[("edge", 3)],
            image=image,
        )


class WangTemplate3C(WangTemplate):
    def __init__(
            self,
            image: torch.Tensor,
    ):
        super().__init__(
            indices=OPTIMAL_WANG_INDICES_SQUARE[("corner", 3)],
            image=image,
        )
