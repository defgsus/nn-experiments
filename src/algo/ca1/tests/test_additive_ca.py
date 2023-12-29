import torch

from tests.base import *

from src.algo import ca1


class TestAdditiveCa1(TestBase):

    def test_100_rule(self):
        rules = ca1.Ca1AdditiveRules(num_states=2, num_neighbours=1)
        self.assertEqual(8, len(rules))
        self.assertEqual([0, 0, 0], rules.kernel(0).tolist())
        self.assertEqual([1, 0, 0], rules.kernel(1).tolist())
        self.assertEqual([0, 1, 0], rules.kernel(2).tolist())
        self.assertEqual([1, 1, 0], rules.kernel(3).tolist())
        self.assertEqual([0, 0, 1], rules.kernel(4).tolist())
        self.assertEqual([1, 0, 1], rules.kernel(5).tolist())
        self.assertEqual([0, 1, 1], rules.kernel(6).tolist())
        self.assertEqual([1, 1, 1], rules.kernel(7).tolist())

    def test_101_rule(self):
        rules = ca1.Ca1AdditiveRules(num_states=3, num_neighbours=1)
        self.assertEqual(27, len(rules))
        self.assertEqual([0, 0, 0], rules.kernel(0).tolist())
        self.assertEqual([1, 0, 0], rules.kernel(1).tolist())
        self.assertEqual([2, 0, 0], rules.kernel(2).tolist())
        self.assertEqual([0, 1, 0], rules.kernel(3).tolist())
        self.assertEqual([1, 1, 0], rules.kernel(4).tolist())
        self.assertEqual([2, 1, 0], rules.kernel(5).tolist())
        self.assertEqual([0, 2, 0], rules.kernel(6).tolist())
        self.assertEqual([1, 2, 0], rules.kernel(7).tolist())
        self.assertEqual([2, 2, 0], rules.kernel(8).tolist())
        self.assertEqual([0, 0, 1], rules.kernel(9).tolist())

    def test_102_rule(self):
        rules = ca1.Ca1AdditiveRules(num_states=2, num_neighbours=2)
        self.assertEqual(32, len(rules))
        self.assertEqual([0, 0, 0, 0, 0], rules.kernel(0).tolist())
        self.assertEqual([1, 0, 0, 0, 0], rules.kernel(1).tolist())
        self.assertEqual([0, 1, 0, 0, 0], rules.kernel(2).tolist())
        self.assertEqual([1, 1, 0, 0, 0], rules.kernel(3).tolist())
        self.assertEqual([0, 0, 1, 0, 0], rules.kernel(4).tolist())
        self.assertEqual([1, 0, 1, 0, 0], rules.kernel(5).tolist())
        self.assertEqual([0, 1, 1, 0, 0], rules.kernel(6).tolist())
        self.assertEqual([1, 1, 1, 0, 0], rules.kernel(7).tolist())
        self.assertEqual([0, 0, 0, 1, 0], rules.kernel(8).tolist())
        self.assertEqual([1, 0, 0, 1, 0], rules.kernel(9).tolist())

    def test_200_run(self):
        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
            ]).to(torch.uint8),
            ca1.ca1_additive_step(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                kernel=ca1.Ca1AdditiveRules(num_states=2, num_neighbours=1).kernel(5),
                num_states=2,
                iterations=4,
                wrap=False,
            ),
        )

        self.assertTensorEqual(
            torch.Tensor([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
            ]).to(torch.uint8),
            ca1.ca1_additive_step(
                input=torch.Tensor([0, 0, 0, 1, 0, 0, 0]).to(torch.uint8),
                kernel=ca1.Ca1AdditiveRules(num_states=2, num_neighbours=1).kernel(5),
                num_states=2,
                iterations=9,
                wrap=True,
            ),
        )

