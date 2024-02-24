import random
from typing import Tuple, Optional, Union, Iterable, Callable

import torch
from torch.utils.data import Dataset, IterableDataset

from .base_iterable import BaseIterableDataset


class ContrastiveIterableDataset(BaseIterableDataset):
    """
    Expects source dataset that generates (vector, id)
    and yields (vector1, vector2, is_similar).
    """
    def __init__(
            self,
            ds: Union[Dataset, IterableDataset],
            contrastive_ratio: float = .5,
            transform_ratio: float = 0.,
            transforms: Optional[Iterable[Callable[[torch.Tensor, torch.Tensor, bool], Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ):
        """
        :param ds: Source dataset
            Expected to yield tuples of (vector, id).
            The id must be int or str convertible!

        :param contrastive_ratio: float (0, 1)
            defines the ratio of dissimilar vectors that are yielded.

            Internally, the source vectors are stored in memory until
            enough are available to yield the required combinations.
            Do not set the value to 0.0 or 1.0, this would uselessly
            require a lot of memory.
        """
        super().__init__()
        self.ds = ds
        self.contrastive_ratio = contrastive_ratio
        self.transform_ratio = transform_ratio
        self.transforms = None if transforms is None else list(transforms)
        self._num_all = 0
        self._num_contrastive = 0
        self._item_map = {}

    def __iter__(self):
        self._num_all = 0
        self._num_contrastive = 0
        self._item_map = {}

        for item in self.ds:
            vector, id = item[:2]
            id = int(id) if isinstance(id, torch.Tensor) else str(id)
            if id not in self._item_map:
                self._item_map[id] = []
            self._item_map[id].append(vector)

            yield from self._yield_items(self._item_map, follow_rules=True)

        while len(self._item_map) >= 2:
            yield from self._yield_items(self._item_map, follow_rules=False)

    def _yield_items(self, item_map, follow_rules: bool):
        #print(len(item_map))
        if len(item_map) >= 2:

            ids = list(item_map.keys())
            id1 = random.choice(ids)
            id2 = None
            is_same = False

            ratio = self._num_contrastive / max(1, self._num_all)

            if ratio > self.contrastive_ratio:
                if len(item_map[id1]) >= 2:
                    id2 = id1
                    is_same = True

            else:
                while True:
                    id2 = random.choice(ids)
                    if id2 != id1:
                        break

            if id2 is None:
                if follow_rules:
                    return

                else:
                    while True:
                        id2 = random.choice(ids)
                        if id2 != id1 or len(item_map[id1]) >= 2:
                            break

            vector1 = item_map[id1].pop(0)
            vector2 = item_map[id2].pop(0)

            for id in (id1, id2):
                if id in item_map and not item_map[id]:
                    del item_map[id]

            if self.transforms is not None and self.transform_ratio and random.random() < self.transform_ratio:
                for t in self.transforms:
                    vector1, vector2 = t(vector1, vector2, is_same)

            yield vector1, vector2, is_same

            self._num_all += 1
            if not is_same:
                self._num_contrastive += 1

