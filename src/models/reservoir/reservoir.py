from typing import Union, Tuple, Optional, Callable

import torch
import torch.nn as nn

from src.models.util import activation_to_callable


def check_valid_reservoir(reservoir: nn.Module):
    assert hasattr(reservoir, "num_inputs"), f"reservoir {reservoir} needs `num_inputs` attribute"
    assert hasattr(reservoir, "num_states"), f"reservoir {reservoir} needs `num_states` attribute"


class Reservoir(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_cells: int,
            leak_rate: Union[float, Tuple[float, float]] = .5,
            rec_std: float = 1.,
            rec_prob: float = .5,
            input_std: float = 1.,
            input_prob: float = .5,
            activation: Union[str, Callable] = "tanh",
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_cells = self.num_states = num_cells
        if isinstance(leak_rate, (int, float)):
            self.leak_rate = leak_rate
        else:
            self.leak_rate = nn.Parameter(torch.rand(self.num_cells) * (leak_rate[1] - leak_rate[0]) + leak_rate[0])

        self.activation = activation_to_callable(activation)
        self.bias_recurrent = nn.Parameter(torch.randn(self.num_cells))
        self.weight_recurrent = nn.Parameter(
            torch.randn(self.num_cells, self.num_cells) * (torch.rand(self.num_cells, self.num_cells) < rec_prob) * rec_std
        )
        self.weight_input = nn.Parameter(
            torch.randn(self.num_inputs, self.num_cells) * (torch.rand(self.num_inputs, self.num_cells) < input_prob) * input_std
        )

    def extra_repr(self) -> str:
        return f"num_inputs={self.num_inputs}, num_cells={self.num_cells}"

    def forward(self, state: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert state.ndim == 2, f"Expected `state` to have shape (B, num_state), got {state.shape}"
        assert state.shape[-1] == self.num_states, \
            f"Expected final dimension of `state` to match `num_states` {self.num_states}, got {state.shape}"

        rec_state = (state + self.bias_recurrent) @ self.weight_recurrent
        rec_state = self.activation(rec_state)

        if input is not None:
            assert input.ndim == 2, f"Expecting `input` to have shape (B, num_inputs), got {input.shape}"
            assert input.shape[-1] == self.num_inputs, \
                f"Expected final dimension of `input` to match `num_inputs` {self.num_inputs}, got {input.shape}"

            in_state = input @ self.weight_input
            rec_state = rec_state + self.activation(in_state)

        next_state = state * (1. - self.leak_rate) + rec_state * self.leak_rate
        return next_state
