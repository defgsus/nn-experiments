import unittest

import torch
import torch.nn as nn

from src.tests.base import *

from src.models.ca import TotalCALayer


class TestTotalCA(TestBase):

    def test_100_classic_gol(self):
        model = TotalCALayer(
            birth=  (0, 0, 0, 1, 0, 0, 0, 0, 0),
            survive=(0, 0, 1, 1, 0, 0, 0, 0, 0),
            iterations=1,
        )
        state_sequence = [
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
        input = torch.Tensor(state_sequence[0])

        for expected_next_state in state_sequence[1:]:
            next_state = model(input.unsqueeze(0))[0]
            self.assertTensorEqual(
                expected_next_state,
                next_state,
            )
            input = next_state

    # There is no gradient flowing back..
    @unittest.expectedFailure
    def test_200_module_autograd(self):
        model = TotalCALayer(
            birth=  (0, 0, 0, 1, 0, 0, 0, 0, 0),
            survive=(0, 0, 1, 1, 0, 0, 0, 0, 0),
            iterations=1,
        )
        data = nn.Parameter(
            torch.rand((1, 32, 32)),
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([data], 0.01)

        grad_max = 0.
        for i in range(10):
            output = model(data)
            loss = (.5 - output).mean()
            loss.backward()
            optimizer.step()
            grad_max = max(grad_max, data.grad.max().item())

        self.assertGreater(grad_max, 0.)

