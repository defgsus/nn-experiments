import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F


class RecursiveUnit(torch.nn.Module):
    def __init__(
            self,
            n_cells: int,
            act_fn: Optional[Callable] = None,
            init_weights: bool = False,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.act_fn = act_fn
        self.recursive_weights = torch.nn.Parameter(
            torch.empty((self.n_cells, self.n_cells))
        )
        if init_weights:
            self.init_weights()

    def init_weights(self, self_connect: bool = False):
        with torch.no_grad():
            self.recursive_weights[:] = torch.randn((self.n_cells, self.n_cells)) / math.sqrt(self.n_cells)
            if not self_connect:
                self.recursive_weights *= (1. - torch.diagflat(torch.Tensor([1] * self.n_cells)))

    def forward(
            self,
            x: torch.Tensor,
            n_iter: int = 1,
    ) -> torch.Tensor:
        for i in range(n_iter):
            x = self.single_pass(x)
        return x

    def forward_history(
            self,
            x: torch.Tensor,
            n_iter: int = 1,
            start_iter: int = 0,
    ) -> torch.Tensor:
        assert start_iter < n_iter, f"start_iter={start_iter} n_iter={n_iter}"

        history = []
        for i in range(n_iter):
            x = self.single_pass(x)
            if i >= start_iter:
                history.append(x.unsqueeze(0))

        return torch.cat(history) if history else torch.Tensor()

    def single_pass(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.recursive_weights
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def threshold(self, x: torch.Tensor, threshold: float, on: float = 1., off: float = 0.) -> torch.Tensor:
        #return torch.where(x >= threshold, on, off)
        return torch.sigmoid((x - threshold) * 10000)

    def calc_hebbian_similarity(
            self,
            x: torch.Tensor,
            n_iter: int = 1,
            threshold: float = .5,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.view(1, -1)

        state_0 = self.single_pass(x)
        state_1 = state_0
        for i in range(n_iter):
            state_1 = self.single_pass(state_1)

        state_0 = self.threshold(state_0, threshold)
        state_1 = self.threshold(state_1, threshold)
        distance = torch.abs(state_1 - state_0)

        similarity = 1. - distance.mean(axis=0)
        similarity = similarity.repeat(similarity.shape[0], 1)
        similarity = similarity * similarity.T
        return similarity

    def hebbian_loss(
            self,
            x: torch.Tensor,
            n_iter: int = 1,
            threshold: float = .5,
    ) -> torch.Tensor:
        similarity = self.calc_hebbian_similarity(x, n_iter=n_iter, threshold=threshold)

        weight_adjust = (similarity - .5) - self.recursive_weights
        #weight_adjust = (self.threshold(similarity, .5) - .5) - self.recursive_weights

        return weight_adjust.mean()

    def calc_complexity(
            self,
            x: torch.Tensor,
            n_iter: int = 10,
            start_iter: int = 0,
    ) -> torch.Tensor:
        assert start_iter < n_iter, f"start_iter={start_iter} n_iter={n_iter}"

        if x.ndim == 1:
            x = x.view(1, -1)

        states = self.forward_history(x, n_iter=n_iter, start_iter=start_iter)

        state_max = states.abs().max()
        auto_diffs = []
        for i in range(1, states.shape[0]):
            d = states[i:] - states[:-i]
            auto_diffs.append(d.abs().mean().unsqueeze(0))
        auto_diffs = torch.cat(auto_diffs)
        auto_diffs /= state_max
        return auto_diffs.mean() * auto_diffs.min()

        return torch.where(torch.corrcoef(states).abs() > .7, 1., 0.).mean()

        corr = torch.corrcoef(states)
        return torch.abs(corr).mean() * -.5 + .5
        #return corr[0][1:].mean() * -.5 + .5
