from typing import Union, Tuple, Optional, Callable

import torch
import torch.nn as nn

from .reservoir import check_valid_reservoir


class ParallelReservoirs(nn.Module):
    def __init__(
            self,
            *reservoirs: nn.Module,
            share_inputs: bool = False,
    ):
        super().__init__()
        self.reservoirs = nn.ModuleList()
        self.num_inputs = 0
        self.num_states = 0
        self._input_slices = []
        self._state_slices = []
        for res in reservoirs:
            check_valid_reservoir(res)
            self.reservoirs.append(res)

            if share_inputs:
                self._input_slices.append(slice(0, res.num_inputs))
                self.num_inputs = max(self.num_inputs, res.num_inputs)
            else:
                self._input_slices.append(slice(self.num_inputs, self.num_inputs + res.num_inputs))
                self.num_inputs += res.num_inputs

            self._state_slices.append(slice(self.num_states, self.num_states + res.num_states))

            self.num_states += res.num_states

    def forward(self, state: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert state.ndim == 2, f"Expected `state` to have shape (B, num_state), got {state.shape}"
        assert state.shape[-1] == self.num_states, \
            f"Expected final dimension of `state` to match `num_states` {self.num_states}, got {state.shape}"

        if input is not None:
            assert input.ndim == 2, f"Expecting `input` to have shape (B, num_inputs), got {input.shape}"
            assert input.shape[-1] == self.num_inputs, \
                f"Expected final dimension of `input` to match `num_inputs` {self.num_inputs}, got {input.shape}"

        res_states = [
            state[:, sl]
            for sl in self._state_slices
        ]
        res_inputs = [
            input[:, sl] if input is not None else None
            for sl in self._input_slices
        ]
        result_states = [
            res(res_state, res_input)
            for res, res_state, res_input in zip(self.reservoirs, res_states, res_inputs)
        ]
        return torch.concat(result_states, dim=-1)


class SequentialReservoirs(nn.Module):
    def __init__(
            self,
            *reservoirs: nn.Module,
    ):
        super().__init__()
        self.reservoirs = nn.ModuleList()
        self.num_states = 0
        self._state_slices = []
        for res in reservoirs:
            check_valid_reservoir(res)
            self.reservoirs.append(res)

            self._state_slices.append(slice(self.num_states, self.num_states + res.num_states))

            self.num_states += res.num_states
        self.num_inputs = self.reservoirs[0].num_inputs

    def forward(self, state: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert state.ndim == 2, f"Expected `state` to have shape (B, num_state), got {state.shape}"
        assert state.shape[-1] == self.num_states, \
            f"Expected final dimension of `state` to match `num_states` {self.num_states}, got {state.shape}"

        if input is not None:
            assert input.ndim == 2, f"Expecting `input` to have shape (B, num_inputs), got {input.shape}"
            assert input.shape[-1] == self.num_inputs, \
                f"Expected final dimension of `input` to match `num_inputs` {self.num_inputs}, got {input.shape}"

        res_states = [
            state[:, sl]
            for sl in self._state_slices
        ]
        next_states = []
        for res, res_state in zip(self.reservoirs, res_states):
            next_state = res(res_state, input)
            next_states.append(next_state)
            input = next_state

        return torch.concat(next_states, dim=-1)
