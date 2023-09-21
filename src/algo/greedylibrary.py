from copy import deepcopy
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Dict

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid
from IPython.display import display

from src.util import to_torch_device
from src.util.image import signed_to_image


class GreedyLibrary:
    """
    Collection of library patches of ndim=>1.

    Learns by adjusting the best matching patch to match the input patch
    """
    def __init__(
            self,
            n_entries: int,
            shape: Iterable[int],
            mean: float = 0.,
            std: float = 0.01,
            device: Union[None, str, torch.device] = "cpu",
    ):
        self.device = to_torch_device(device)
        self.shape = tuple(shape)
        self.n_entries = n_entries
        self.entries = mean + std * torch.randn(n_entries, *self.shape).to(self.device)
        self.hits = [0] * n_entries

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_entries}, {self.shape})"

    def __copy__(self):
        return self.copy()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def max_hits(self) -> int:
        """Maximum number of hits of all entries"""
        return max(0, *self.hits) if self.hits else 0

    def copy(self) -> "GreedyLibrary":
        d = self.__class__(0, self.shape, device=self.device)
        d.n_entries = self.n_entries
        d.entries = deepcopy(self.entries)
        d.hits = self.hits.copy()
        return d

    def to(self, device: Union[str, torch.device], inplace: bool = False) -> "GreedyLibrary":
        if not inplace:
            lib = self.copy()
            return lib.to(device, inplace=True)

        self.device = to_torch_device(device)
        self.entries = self.entries.to(self.device)
        return self

    def cpu(self, inplace: bool = False) -> "GreedyLibrary":
        return self.to("cpu", inplace=inplace)

    def cuda(self, inplace: bool = False) -> "GreedyLibrary":
        return self.to("cuda", inplace=inplace)

    def save_torch(self, f: torch.serialization.FILE_LIKE, **kwargs):
        torch.save(self._save_data(), f, **kwargs)

    def load_torch(self, f: torch.serialization.FILE_LIKE, **kwargs):
        data = torch.load(f, **kwargs)
        self._load_data(data)

    @classmethod
    def from_torch(cls, f: torch.serialization.FILE_LIKE, device: Union[None, str, torch.device] = "cpu") -> "GreedyLibrary":
        lib = cls(n_entries=0, shape=tuple(), device=device)
        lib.load_torch(f)
        return lib

    def _save_data(self) -> dict:
        return {
            "entries": self.entries,
            "hits": self.hits,
        }

    def _load_data(self, data: dict):
        self.entries = data["entries"]
        self.hits = data["hits"]
        self.n_entries = len(self.hits)
        self.shape = tuple(self.entries.shape[1:])
        self.to(self.device)

    def top_entry_index(self) -> Optional[int]:
        """Returns index of entry with most hits"""
        top_idx, top_hits = None, None
        for i, hits in enumerate(self.hits):
            if top_idx is None or hits > top_hits:
                top_idx, top_hits = i, hits
        return top_idx

    def entry_ranks(self, reverse: bool = False) -> List[int]:
        """
        Returns a list of ranks for each entry,
        where rank means the index sorted by number of hits.
        """
        entry_ids = list(range(self.n_entries))
        entry_ids.sort(key=lambda i: self.hits[i], reverse=reverse)
        return [entry_ids.index(i) for i in range(self.n_entries)]

    def entry_hits(self, reverse: bool = False) -> Dict[int, int]:
        """
        Returns a dict of `entry-index` -> `number-of-hits`.

        Sorted by number of hits.
        """
        entry_ids = list(range(self.n_entries))
        entry_ids.sort(key=lambda i: self.hits[i], reverse=reverse)
        return {
            i: self.hits[i]
            for i in entry_ids
        }

    def sort_entries(
            self,
            by: str = "hits",
            reverse: bool = False,
            inplace: bool = False,
    ):
        if not inplace:
            lib = self.copy()
            return lib.sort_entries(by=by, reverse=reverse, inplace=True)

        sorted_ids = self.sorted_entry_indices(by=by, reverse=reverse)
        self.entries = self.entries[sorted_ids]
        self.hits = [self.hits[sorted_ids[i]] for i in range(len(self.hits))]
        return self

    def sorted_entry_indices(
            self,
            by: str = "hits",
            reverse: bool = False,
    ) -> List[int]:
        if not self.n_entries:
            return []
        entry_ids = list(range(self.n_entries))
        if self.n_entries < 2:
            return entry_ids

        if by == "hits":
            entry_ids.sort(key=lambda i: self.hits[i], reverse=reverse)

        elif by == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(1, perplexity=min(30, self.n_entries - 1))
            positions = tsne.fit_transform(self.entries.reshape(self.entries.shape[0], -1).cpu().numpy())
            entry_ids.sort(key=lambda i: positions[i], reverse=reverse)

        else:
            raise ValueError(f"Unsupported sort by '{by}'")

        return entry_ids

    def fit(
            self,
            batch: torch.Tensor,
            lr: float = 1.,
            metric: str = "l1",
            zero_mean: bool = False,
            skip_top_entries: Union[bool, int] = False,
            grow_if_distance_above: Optional[float] = None,
            max_entries: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Partially fit a batch of patches.

        :param batch: Tensor of N patches of shape matching the library's shape
        :param lr: learning rate, range [0, 1]
        :param zero_mean: True to subtract the mean from each patch in the batch
        :param skip_top_entries: bool or int,
            Do not match the top N entries (1 for True), sorted by number of hits
        :return: tuple of
            - Tensor of int64: entry indices
            - Tensor of float: distances
        """
        batch = batch.to(self.device)

        if zero_mean:
            batch_mean = batch
            for i in range(self.ndim):
                batch_mean = batch_mean.mean(dim=i+1, keepdim=True)
            batch = batch - batch_mean

        best_entry_ids, distances = self.best_entries_for(
            batch, skip_top_entries=skip_top_entries, metric=metric
        )

        for i in range(batch.shape[0]):
            entry_id = best_entry_ids[i]

            if grow_if_distance_above is not None:
                if distances[i] > grow_if_distance_above and self.n_entries < max_entries:
                    self.entries = torch.concat([
                        self.entries,
                        torch.randn(1, *self.shape).to(self.device) * 0.001 + self.entries.mean()
                    ])
                    self.hits.append(0)
                    entry_id = self.n_entries
                    self.n_entries += 1

            weight = 1. / (1 + self.hits[entry_id])
            self.entries[entry_id] += lr * weight * (batch[i] - self.entries[entry_id])
            self.hits[entry_id] += 1

        return best_entry_ids, distances

    def convolve(
            self,
            x: torch.Tensor,
            stride: Union[int, Iterable[int]] = 1,
            padding: Union[int, Iterable[int]] = 0,
    ) -> torch.Tensor:
        func = getattr(F, f"conv{self.ndim - 1}d", None)
        if not callable(func):
            raise NotImplementedError(f"{self.ndim - 1}-d convolution not supported")

        return func(x.to(self.device), self.entries, stride=stride, padding=padding)

    def best_entries_for(
            self,
            batch: torch.Tensor,
            skip_top_entries: Union[bool, int] = False,
            metric: str = "l1",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the index of the best matching entry for each patch in the batch.

        :param batch: Tensor of N patches of shape matching the library's shape
        :param skip_top_entries: bool or int,
            Do not match the top N entries (1 for True), sorted by number of hits
        :return: tuple of
            - Tensor of int64: entry indices
            - Tensor of float: distances
        """
        assert batch.ndim == len(self.shape) + 1, f"Got {batch.shape}"
        assert batch.shape[1:] == self.shape, f"Got {batch.shape}"
        ones = tuple(1 for _ in self.shape)

        repeated_entries = self.entries.repeat(batch.shape[0], *ones)
        repeated_batch = batch.to(self.device).repeat(1, self.n_entries, *ones[1:]).reshape(-1, *self.shape)

        if metric in ("l1", "mae"):
            dist = (repeated_entries - repeated_batch).abs().flatten(1).sum(1)
        elif metric in ("l2", "mse"):
            dist = (repeated_entries - repeated_batch).pow(2).flatten(1).sum(1).sqrt()
        elif metric.startswith("corr"):
            dist = -(repeated_entries * repeated_batch).flatten(1).sum(1)

        dist = dist.reshape(batch.shape[0], -1)

        if not skip_top_entries:
            indices = torch.argmin(dist, 1)
            return (
                indices,
                dist.flatten()[
                    indices + torch.linspace(0, indices.shape[0] - 1, indices.shape[0]).to(torch.int64).to(indices.device) * self.n_entries
                ]
            )

        skip_top_entries = int(skip_top_entries)
        sorted_indices = torch.argsort(dist, 1)
        entry_ranks = self.entry_ranks(reverse=True)
        best_entries = []
        for indices in sorted_indices:
            idx = 0
            while idx + 1 < len(indices) and entry_ranks[indices[idx]] < skip_top_entries:
                idx += 1
            best_entries.append(indices[idx])

        indices = torch.Tensor(best_entries).to(torch.int64)
        return (
            indices,
            dist.flatten()[
                indices + torch.linspace(0, indices.shape[0] - 1, indices.shape[0]).to(torch.int64).to(indices.device) * self.n_entries
            ]
        )

    def drop_unused(self, inplace: bool = False) -> "GreedyLibrary":
        return self.drop_entries(hits_lt=1, inplace=inplace)

    def drop_entries(
            self,
            hits_lt: Optional[int] = None,
            inplace: bool = False,
    ) -> "GreedyLibrary":
        if not inplace:
            lib = self.copy()
            return lib.drop_entries(
                hits_lt=hits_lt,
                inplace=True,
            )

        drop_idx = set()
        if hits_lt is not None:
            for i, hits in enumerate(self.hits):
                if hits <= hits_lt:
                    drop_idx.add(i)

        if drop_idx:
            entries = []
            hits = []
            for i, (entry, h) in enumerate(zip(self.entries, self.hits)):
                if i not in drop_idx:
                    entries.append(entry.unsqueeze(0))
                    hits.append(h)
            self.entries = torch.concat(entries) if entries else torch.Tensor()
            self.hits = hits
            self.n_entries = len(self.hits)
            self.to(self.device, inplace=True)

        return self

    def plot_entries(
            self,
            min_size: int = 300,
            with_hits: bool = True,
            sort_by: Optional[str] = None,
            signed: bool = False,
    ) -> PIL.Image.Image:
        if len(self.shape) == 1:
            entries = self.entries.reshape(-1, 1, 1, *self.shape)
        elif len(self.shape) == 2:
            entries = self.entries.reshape(-1, 1, *self.shape)
        elif len(self.shape) == 3:
            entries = self.entries
        else:
            raise RuntimeError(f"Can't plot entries with shape {self.shape} (ndim>3)")

        if entries.shape[0]:

            if not signed:
                # normalize
                e_min, e_max = entries.min(), entries.max()
                if e_min != e_max:
                    entries = (entries - e_min) / (e_max - e_min)

            else:
                entries = torch.concat([signed_to_image(e, normalize=False).unsqueeze(0) for e in entries])
                max_value = entries.max()
                if max_value:
                    entries /= max_value

            if with_hits:
                max_hits = max(1, self.max_hits)
                entry_list = []
                for entry, hits in zip(entries, self.hits):
                    if entry.shape[0] == 1:
                        entry = entry.repeat(3, *(1 for _ in entry.shape[1:]))
                    elif entry.shape[0] == 3:
                        pass
                    else:
                        raise ValueError(f"Can't plot entries with {entry.shape[0]} channels")

                    background = torch.Tensor([0, hits / max_hits, 0])
                    background = background.reshape(3, *((1,) * (len(entry.shape) - 1)))
                    background = background.repeat(1, *(s + 2 for s in entry.shape[1:]))
                    background[:, 1:-1, 1:-1] = entry
                    entry_list.append(background)
                entries = entry_list

            if sort_by:
                if not isinstance(entries, list):
                    entries = list(entries)
                entry_ids = self.sorted_entry_indices(by=sort_by, reverse=True)
                entries = [entries[i] for i in entry_ids]

            grid = make_grid(entries, nrow=max(1, int(np.sqrt(self.n_entries))), normalize=False)
            if grid.shape[-1] < min_size:
                grid = VF.resize(
                    grid,
                    [
                        int(grid.shape[-2] * min_size / grid.shape[-1]),
                        min_size,
                    ],
                    VF.InterpolationMode.NEAREST
                )
        else:
            grid = torch.zeros(1, min_size, min_size)
        return VF.to_pil_image(grid.cpu())
