import unittest
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.functional import *
from tests.base import *


class TestSoftHistogram(TestBase):

    def test_100_shape_flat(self):
        for bins in (1, 10, 100):
            self.assertEqual(
                torch.Size((bins,)),
                soft_histogram_flat(torch.rand(2342), bins, 0, 1).shape,
            )
            self.assertEqual(
                torch.Size((bins,)),
                soft_histogram_flat(torch.rand(1, 2342), bins, 0, 1).shape,
            )
            self.assertEqual(
                torch.Size((bins,)),
                soft_histogram_flat(torch.rand(23, 155), bins, 0, 1).shape,
            )

    def test_110_shape_batched(self):
        for bins in (1, 10, 100):
            self.assertEqual(
                torch.Size((1, bins,)),
                soft_histogram(torch.rand(2342), bins, 0, 1).shape,
            )
            self.assertEqual(
                torch.Size((1, bins,)),
                soft_histogram(torch.rand(1, 2342), bins, 0, 1).shape,
            )
            self.assertEqual(
                torch.Size((23, bins,)),
                soft_histogram(torch.rand(23, 155), bins, 0, 1).shape,
            )

    def test_200_compare_flat_with_torch(self):
        for bins in (10, 100, ):
            for i in range(10):
                while True:
                    data = self.get_random_data((1, 10000))[0]

                    t_hist = torch.histc(data, bins, -1, 1)
                    if t_hist.max() >= 1:
                        break

                s_hist = soft_histogram_flat(data, bins, -1, 1, sigma=100_000)

                ma = t_hist.max()
                error = F.l1_loss(s_hist / ma, t_hist / ma)
                # print(error, t_hist.max())

                self.assertLess(error, .01, f"For bins={bins}")

    def test_210_compare_batch_with_torch(self):
        for bins in (10, 100, ):
            for i in range(10):
                while True:
                    data = self.get_random_data((3, 10000))

                    t_hist = torch.concat([
                        torch.histc(d, bins, -1, 1).unsqueeze(0)
                        for d in data
                    ])
                    if torch.all(t_hist.max(dim=-1)[0]) >= 1:
                        break

                s_hist = soft_histogram(data, bins, -1, 1, sigma=100_000)

                ma = t_hist.max()
                error = F.l1_loss(s_hist / ma, t_hist / ma)
                # print(error, t_hist.max())

                self.assertLess(error, .01, f"For bins={bins}, max={t_hist.max()}")

    def get_random_data(self, shape: Tuple[int, int]):
        t = torch.linspace(0, 2 * torch.pi * shape[-1], shape[-1])
        t = t.unsqueeze(0).expand(shape[0], -1)
        data = torch.randn_like(t) * .3

        for i in range(4):
            f = torch.randn(3, shape[0], 1)
            f *= torch.randn_like(f)
            data += torch.tanh(torch.sin(t * 7. * f[0]) * f[1] * .3 + f[2])
        return data

    def test_500_is_differentiable(self):
        num_samples = 1000
        epochs = 500

        data_batch = torch.rand(num_samples, 32)
        target_batch = torch.ones(num_samples, 10)

        model = nn.Sequential(
            nn.Linear(32, 100),
            nn.Linear(100, 10),
            nn.Sigmoid(),
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []
        with tqdm(total=epochs) as progress:
            for epoch in range(epochs):
                output_batch = model(data_batch)

                output_hist_batch = soft_histogram(output_batch, target_batch.shape[-1], 0, 1)[0].unsqueeze(0)

                #print(output_batch.mean())
                #print(output_hist_batch[0])
                loss = F.l1_loss(output_hist_batch, target_batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss = round(float(loss), 2)
                losses.append(loss)

                progress.desc = f"training, loss={loss}"
                progress.update(1)

        self.assertLess(losses[-1], losses[0], f"No loss reduction, training failed!\nLosses: {losses}")
        print(losses)
