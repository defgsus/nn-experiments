import unittest
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.functional import differentiable_histogram
from src.tests.base import *


class TestNaiveHistogram(TestBase):

    def test_100_compare_with_torch_histogram(self):
        for bins in (
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100,
                [0, .1, .4, .9, 1.2], [-1, 0.1],
        ):
            if isinstance(bins, list):
                bins = torch.Tensor(bins)

            for values in (
                    [0], [1], [0, 1], [0, .1], [-.1, 0], [-1, 4],
                    torch.randn(100), torch.rand(1, 2, 3, 100) * 17.,
            ):
                for range in (
                        None, (0, 1), (-1, 1), (0.01, 0.02), (-5, 5), (.5, 13.77)
                ):
                    self.compare_with_torch_histogram(
                        torch.Tensor(values),
                        bins=bins,
                        range=range,
                    )

    # it's not really differentiable right now
    @unittest.expectedFailure
    def test_200_is_differentiable(self):
        num_samples = 2
        epochs = 500

        data_batch = torch.rand(num_samples, 32)
        target_batch = torch.ones(num_samples, 10)

        model = nn.Sequential(
            nn.Linear(32, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 10),
            nn.Sigmoid(),
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        with tqdm(total=epochs) as progress:
            for epoch in range(epochs):
                output_batch = model(data_batch)

                output_hist_batch = torch.concat([
                    differentiable_histogram(out, bins=target_batch.shape[-1], range=(0, 1))[0].unsqueeze(0)
                    for out in output_batch
                ])

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

    def compare_with_torch_histogram(
            self,
            x: torch.Tensor,
            bins: Union[int, torch.Tensor],
            *,
            range: Optional[Tuple[float, float]] = None,
    ):
        msg = f"x={x}, bins={bins}, range={range}"

        expect_exception = None

        try:
            if range is None:
                h1 = torch.histogram(x, bins=bins)
            else:
                h1 = torch.histogram(x, bins=bins, range=range)
        except Exception as e:
            expect_exception = type(e)

        if expect_exception is None:
            try:
                h2 = differentiable_histogram(x, bins=bins, range=range)
            except Exception as e:
                raise type(e)(f"{e}. This exception was unexpected for {msg}")

            self.assertTensorEqual(
                h1[1],
                h2[1],
                f"Bins do not match for {msg}"
            )
            self.assertTensorEqual(
                h1[0],
                h2[0],
                f"Histograms do not match for {msg}"
            )

        else:
            with self.assertRaises(expect_exception, msg=msg):
                differentiable_histogram(x, bins=bins, range=range)

