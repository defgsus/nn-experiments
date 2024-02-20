import unittest

import torch
import torch.nn as nn

from src.tests.base import *

from src.algo import ca1


class TestReplaceCa1(TestBase):

    def test_100_rule(self):
        rules = ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1)
        self.assertEqual(256, len(rules))
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(0).tolist())
        self.assertEqual([1, 0, 0, 0, 0, 0, 0, 0], rules.lookup(1).tolist())
        self.assertEqual([0, 1, 0, 0, 0, 0, 0, 0], rules.lookup(2).tolist())
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], rules.lookup(3).tolist())
        self.assertEqual([0, 1, 1, 1, 1, 0, 0, 0], rules.lookup(30).tolist())

    def test_101_rule(self):
        rules = ca1.Ca1ReplaceRules(num_states=3, num_neighbours=1)
        self.assertEqual(7625597484987, len(rules))  # 3^3^3
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(0).tolist())
        self.assertEqual([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(1).tolist())
        self.assertEqual([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(2).tolist())
        self.assertEqual([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(3).tolist())
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(4).tolist())
        self.assertEqual([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(5).tolist())
        self.assertEqual([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(6).tolist())
        self.assertEqual([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(7).tolist())
        self.assertEqual([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(8).tolist())

    def test_102_rule(self):
        rules = ca1.Ca1ReplaceRules(num_states=2, num_neighbours=2)
        self.assertEqual(4294967296, len(rules))  # 2^2^5
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(0).tolist())
        self.assertEqual([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(1).tolist())
        self.assertEqual([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(2).tolist())
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(3).tolist())
        self.assertEqual([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], rules.lookup(4).tolist())

    def test_200_run(self):
        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 1, 0],
                [1, 1, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1],
            ]).to(torch.uint8),
            ca1.ca1_replace_step(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(30),
                num_states=2,
                num_neighbours=1,
                iterations=6,
                wrap=False,
            ),
        )

        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 1, 0],
                [1, 1, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 1, 0]
            ]).to(torch.uint8),
            ca1.ca1_replace_step(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(30),
                num_states=2,
                num_neighbours=1,
                iterations=6,
                wrap=True,
            ),
        )

        if 1:
            steps = 50
            input = torch.zeros(steps, dtype=torch.uint8)
            input[input.shape[-1] // 2] = 1
            state = ca1.ca1_replace_step(
                input=input,
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(30),
                num_states=2,
                num_neighbours=1,
                iterations=steps,
                wrap=True,
            )
            print()
            for row in state:
                print("".join(
                    "X" if c else "." for c in row
                ))

    def test_210_run_batched(self):
        self.assertTensorEqual(
            torch.Tensor([
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 1],
                    [1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0, 1]
                ]
            ]).to(torch.uint8),
            ca1.ca1_replace_step(
                input=torch.Tensor([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]).to(torch.uint8),
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(30),
                num_states=2,
                num_neighbours=1,
                iterations=6,
            ),
        )

    def test_250_run_seq_input(self):
        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
            ]).to(torch.uint8),
            ca1.ca1_replace_step(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(4),
                num_neighbours=1,
                num_states=2,
                iterations=6,
                wrap=False,
                seq_input=torch.Tensor([
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                ]).to(torch.uint8)
            ),
        )
        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
            ]).to(torch.uint8),
            ca1.ca1_replace_step(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(4),
                num_neighbours=1,
                num_states=2,
                iterations=6,
                wrap=False,
                seq_input=torch.Tensor([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                ]).to(torch.uint8),
                seq_input_stride=2,
            ),
        )

    def test_260_run_seq_input_batched(self):
        self.assertTensorEqual(
            torch.Tensor([
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1],
                ]
            ]).to(torch.uint8),
            ca1.ca1_replace_step(
                input=torch.Tensor([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]).to(torch.uint8),
                lookup=ca1.Ca1ReplaceRules(num_states=2, num_neighbours=1).lookup(4),
                num_states=2,
                num_neighbours=1,
                iterations=6,
                seq_input=torch.Tensor([
                    [
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0],
                    ]
                ]).to(torch.uint8),
            ),
        )

    def test_300_fuzzing(self):
        for num_states in (2, 3):
            for num_neighbours in (1, 2, 3):
                for input_size in (1, 2, 3, 4, 5, 6, 7, 8):
                    for wrap in (False, True):
                        try:
                            ca1.ca1_replace_step(
                                input=torch.linspace(0, input_size - 1, input_size, dtype=torch.uint8) % num_states,
                                lookup=ca1.Ca1ReplaceRules(num_states=num_states, num_neighbours=num_neighbours).lookup(30),
                                num_states=num_states,
                                num_neighbours=num_neighbours,
                                iterations=10,
                                wrap=wrap,
                            )
                        except Exception as e:
                            e.args += (
                                f"num_states={num_states}, num_n={num_neighbours}, input_size={input_size}, wrap={wrap}",
                            )
                            raise e

    def test_400_module(self):
        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 1, 0],
                [1, 1, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 1],
            ]).to(torch.uint8),
            ca1.CA1Replace(rule=30)(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                iterations=6,
                wrap=False,
            ),
        )

    # this is currently not working
    @unittest.expectedFailure
    def test_410_module_autograd(self):
        data = nn.Parameter(
            torch.Tensor([0, 0, 0, 1, 0, 0, 0]),
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([data])

        module = ca1.CA1Replace(rule=30)

        output = module(
            input=data,
            iterations=6,
            wrap=False,
        )
        loss = output.sum()
        loss.backward()
        optimizer.step()

