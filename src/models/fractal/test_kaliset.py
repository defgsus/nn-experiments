import itertools
import random
import unittest

import torch
import torch.nn as nn

from src.tests.base import *

from src.models.fractal import KaliSetLayer


class TestKaliSet(TestBase):

    def test_100_axis(self):
        rng = random.Random(23)
        for ndim in range(1, 5):
            for num_param in range(2, 6):
                for axis in itertools.chain(range(0, ndim)):
                    for scale in (None, (.1, ) * num_param):
                        for offset in (None, (.1, ) * num_param):
                            for iter_random in range(4):
                                try:
                                    model = KaliSetLayer(
                                        param=torch.linspace(.1, 1., num_param),
                                        axis=axis,
                                        iterations=2,
                                        scale=scale,
                                        offset=offset,
                                    )
                                    shape = [rng.randint(1, 100) for _ in range(ndim)]
                                    shape[axis] = num_param

                                    input = torch.rand(shape)

                                    output = model(input)
                                    self.assertEqual(input.shape, output.shape)

                                except:
                                    print(f"PARAMS: ndim={ndim}, num_param={num_param}, axis={axis}")
                                    raise

    def test_200_autograd(self):
        model = KaliSetLayer(
            param=(.5, .5, .5),
            axis=-3,
            iterations=2,
            learn_mixer=True,
            learn_param=True,
            learn_offset=True,
            learn_scale=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), 0.01)
        data = torch.rand(16, 3, 10, 10)

        torch.autograd.set_detect_anomaly(True)
        grad_max = 0.
        for i in range(10):
            output = model(data)
            loss = (.5 - output).mean()
            loss.backward()
            optimizer.step()
            grad_max = max(grad_max, max(p.grad.max().item() for p in model.parameters()))

        self.assertGreater(grad_max, 0.)
