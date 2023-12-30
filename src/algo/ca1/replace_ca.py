from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import pad


class Ca1ReplaceRules:
    def __init__(
            self,
            num_states: int = 2,
            num_neighbours: int = 1,
    ):
        self.num_states = num_states
        self.num_neighbours = num_neighbours
        self.kernel_size = 1 + 2 * num_neighbours
        self.num_rules = num_states ** num_states ** self.kernel_size

    def __len__(self):
        return self.num_rules

    def lookup(self, rule: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        kernel = [
            (rule // (self.num_states ** k) % self.num_states)
            for k in range(self.num_states ** self.kernel_size)
        ]
        return torch.Tensor(kernel).to(dtype)


def ca1_replace_step(
        input: torch.Tensor,
        lookup: torch.Tensor,
        num_neighbours: int,
        iterations: int,
        wrap: bool = False,
) -> torch.Tensor:
    if input.ndim not in (1, 2):
        raise ValueError(f"Expected input.ndim=1 or 2, got {input.shape}")
    if lookup.ndim != 1:
        raise ValueError(f"Expected lookup.ndim=1, got {lookup.shape}")

    width = input.shape[-1]
    n_size = 1 + 2 * num_neighbours             # size of neighbourhood
    width_n = (width // n_size + 1) * n_size    # with rounded up to the next n_size multiple
    # multiplier to get the lookup index for each cell in local neighbourhood
    index_mult = torch.Tensor([2 ** n for n in range(n_size)]).view(1, 1, -1).to(lookup)

    state = input
    if state.ndim == 1:
        state = state.unsqueeze(0)
    batch_size = state.shape[0]

    history = [state.unsqueeze(0)]
    for it in range(iterations):

        padding = (1, n_size + width_n - state.shape[-1] - 1)
        state = pad(state, padding, wrap=wrap)

        index = torch.empty(batch_size, width_n, dtype=torch.int64)
        for k in range(n_size):
            state_slices = state[:, k: width_n + k].view(-1, width_n // n_size, n_size)
            #print(f"k={k}, state={state.shape}, width_n={width_n}, state_slices={state_slices.shape}")
            index_slices = (state_slices * index_mult).sum(dim=-1)

            index[:, k::n_size] = index_slices

        state = torch.index_select(lookup, 0, index.flatten(0)).view(batch_size, -1)
        state = state[:, :width]

        history.append(state.unsqueeze(0))

    state = torch.concat(history)
    if input.ndim == 1:
        state = state.squeeze(1)
    else:
        state = state.permute(1, 0, 2).view(batch_size, -1, state.shape[-1])
    return state


class CA1Replace(nn.Module):

    def __init__(
            self,
            rule: int,
            num_states: int = 2,
            num_neighbours: int = 1,
            wrap: bool = False,
            iterations: Optional[int] = None,
            output_steps: Optional[int] = None,
            dtype: torch.dtype = torch.uint8,
    ):
        super().__init__()
        self.rule = rule
        self.num_states = num_states
        self.num_neighbours = num_neighbours
        self.wrap = wrap
        self.iterations = iterations
        self.output_steps = output_steps
        self.lookup = nn.Parameter(
            Ca1ReplaceRules(num_states=num_states, num_neighbours=num_neighbours).lookup(rule, dtype=dtype),
            requires_grad=False,
        )

    def forward(
            self,
            input: torch.Tensor,
            iterations: Optional[int] = None,
            output_steps: Optional[int] = None,
            wrap: Optional[bool] = None,
            threshold: float = .5,
    ):
        if "int" in str(self.lookup.dtype) and "int" not in str(input.dtype):
            input = (input >= threshold).to(self.lookup.dtype)

        if iterations is None:
            iterations = self.iterations
        if iterations is None:
            raise ValueError(
                f"Must either provide `iterations` in {type(self).__name__} constructor or forward method"
            )
        if output_steps is None:
            output_steps = self.output_steps
        if wrap is None:
            wrap = self.wrap

        state = ca1_replace_step(
            input=input,
            lookup=self.lookup,
            num_neighbours=self.num_neighbours,
            iterations=iterations,
            wrap=wrap,
        )

        if output_steps is not None:
            state = state[..., -output_steps:, :]

        return state
