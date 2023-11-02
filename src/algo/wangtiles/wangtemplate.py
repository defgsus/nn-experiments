from typing import Iterable, Tuple, Optional

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


class WangTemplate2E(WangTemplate):
    def __init__(
            self,
            image: torch.Tensor,
    ):
        super().__init__(
            indices=[
                [4, 6, 14, 12],
                [5, 7, 15, 13],
                [1, 3, 11, 9],
                [0, 2, 10, 8],
            ],
            image=image,
        )


class WangTemplate2C(WangTemplate):
    def __init__(
            self,
            image: torch.Tensor,
    ):
        super().__init__(
            indices=[
                [4,  3, 14, 6],
                [10, 7, 15, 13],
                [1,  9, 11, 12],
                [0,  2, 5,  8],
            ],
            image=image,
        )
