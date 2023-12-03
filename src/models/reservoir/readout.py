import sys
from typing import Union, Tuple, Optional, Callable

import torch
import torch.nn as nn

from sklearn.linear_model import Ridge
from tqdm import tqdm

from .reservoir import check_valid_reservoir

class ReservoirReadout:
    def __init__(
            self,
            reservoir: nn.Module,
            verbose: bool = False,
    ):
        """
        Readout for a "Reservoir Computing" network.

        :param reservoir: nn.Module
            The module needs to provide the following interface:
                - `num_inputs` int parameter
                . `num_states` int parameter
                - A `forward` function that accepts the current state of the reservoir of shape (B, num_states)
                  and optionally a second tensor representing the input of shape (B, num_inputs)
        :param verbose: bool, if True use tqdm for displaying progress
        """
        check_valid_reservoir(reservoir)
        self.reservoir = reservoir
        self.verbose = verbose
        self.ridge = None

    @torch.no_grad()
    def run_reservoir(
            self,
            input: Optional[torch.Tensor] = None,
            state: Optional[torch.Tensor] = None,
            steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run the reservoir for a number of steps

        :param input: optional Tensor of shape (B, T, num_inputs),
            The input to the reservoir for each time-step.
        :param state: optional Tensor of shape (B, num_states),
            The initial state of the reservoir.
            If omitted it will be initialized to zeros.
        :param steps: optional int with the number of steps of the reservoir calculation.
            If omitted it will be either equal to the input time-steps or 1.
            If defined, the size of the input time-steps are ignored. No input is provided
            for calculation steps after `steps`.
        :return: Tensor, the new state of the reservoir
        """
        if input is not None:
            assert input.ndim == 3, f"Expecting `input` of shape (B, T, N), got {input.shape}"
            assert input.shape[-1] == self.reservoir.num_inputs, \
                f"Expecting final dimension of `input` to match `reservior.num_inputs` {self.reservoir.num_inputs}, got {input.shape}"
            batch_size = input.shape[0]
            if steps is None:
                steps = input.shape[1]
        else:
            batch_size = 1

        if state is not None:
            assert state.ndim == 2, f"Expecting `state` of shape (B, C), got {state.shape}"
            assert state.shape[-1] == self.reservoir.num_states, \
                f"Expecting last `state` dimension to match reservoir size {self.reservoir.num_states}, got {state.shape}"
            assert state.shape[0] == batch_size, \
                f"Expecting first `state` dimension to match batch size {batch_size}, got {state.shape}"
        else:
            state = torch.zeros(batch_size, self.reservoir.num_states)

        if steps is None:
            steps = 1

        states = []
        for i in tqdm(range(steps), desc="running reservoir", disable=not self.verbose):
            state = self.reservoir(state, input[:, i] if input is not None and i < input.shape[1] else None)
            states.append(state)

        return torch.concat([s.unsqueeze(1) for s in states], dim=1)

    @torch.no_grad()
    def fit(self, input: torch.Tensor, target: torch.Tensor, alpha: float = 1.) -> Tuple[float, float]:
        """
        Fit the readout to the reservoir.

        This resets any previous training.

        :param input: Tensor, the input to the reservoir in shape (B, T, num_inputs)
        :param target: Tensor, the target values in shape (B, T, X), where X is the number of
            required output elements per time-step.
        :param alpha: float, the `alpha` parameter of the `sklearn.linear_model.Ridge` module.
            It's a mixing parameter for the l2 regularization of the ridge regression, in range [0, inf)
        :return:
        """
        assert input.ndim == 3, f"Expecting input of shape (B, T, N), got {input.shape}"
        assert input.shape[-1] == self.reservoir.num_inputs, \
            f"Expecting final dimension of `input` to match `reservior.num_inputs` {self.reservoir.num_inputs}, got {input.shape}"
        assert target.ndim == 3, f"Expecting target of shape (B, T, N), got {target.shape}"
        assert input.shape[:2] == target.shape[:2], \
            f"Expecting first 2 dimensions of `target` to be equal to `input` {input.shape[:2]}, got {target.shape[:2]}"

        batch_size, time_steps = input.shape[:2]

        state = self.run_reservoir(input=input)

        state = state.reshape(batch_size * time_steps, -1)
        target = target.reshape(batch_size * time_steps, -1)

        if self.verbose:
            print("fitting output...", file=sys.stderr, flush=True)

        self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(state.numpy(), target.numpy())

        prediction = torch.Tensor(self.ridge.predict(state))

        error_l1 = (target - prediction).abs().mean()
        error_l2 = torch.sqrt(((target - prediction) ** 2).sum())
        return float(error_l1), float(error_l2)

    @torch.no_grad()
    def predict(
            self,
            input: Optional[torch.Tensor] = None,
            state: Optional[torch.Tensor] = None,
            steps: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.ridge is not None, "Must call `fit` before `predict`"

        state = self.run_reservoir(input=input, state=state, steps=steps)

        batch_size, time_steps = state.shape[:2]

        state = state.reshape(batch_size * time_steps, -1)

        prediction = torch.Tensor(self.ridge.predict(state))

        return prediction.view(batch_size, time_steps, -1)

    @torch.no_grad()
    def generate(
            self,
            steps: int,
            input: Optional[torch.Tensor] = None,
            state: Optional[torch.Tensor] = None,
            adjust_prediction: Optional[Callable] = None,
            lookahead: int = 1,
    ) -> torch.Tensor:
        state = self.run_reservoir(input=input, state=state)

        batch_size, time_steps = state.shape[:2]

        prediction = torch.Tensor(self.ridge.predict(state.reshape(batch_size * time_steps, -1)))
        prediction = prediction.view(batch_size, time_steps, -1)

        state_slice = state[:, -lookahead]
        # input_slice = prediction[:, -1, :]
        future_predictions = [prediction]
        input_slices = [prediction[:, -i, :] for i in range(lookahead-1, -1, -1)]
        if adjust_prediction is not None:
            input_slices = [adjust_prediction(s) for s in input_slices]

        for i in tqdm(range(steps), desc="generating", disable=not self.verbose):
            state_slice = self.reservoir(state_slice, input_slices.pop(0))
            predicted_slice = torch.Tensor(self.ridge.predict(state_slice))
            if adjust_prediction is not None:
                predicted_slice = adjust_prediction(predicted_slice)
            future_predictions.append(predicted_slice[:, None, :])
            input_slices.append(predicted_slice)

        return torch.concat(future_predictions, dim=-2)
