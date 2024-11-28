import math
import unittest
from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.functional import colorconvert
from src.tests.base import *
from src.util import to_torch_device


class TestColorConvert(TestBase):

    def test_500_is_differentiable(self):
        for method in (
                colorconvert.hsv2rgb,
                colorconvert.rgb2hsv,
                colorconvert.lab2rgb,
                colorconvert.rgb2lab,
                colorconvert.rgb2xyz,
                colorconvert.xyz2rgb,
                colorconvert.xyz2lab,
                colorconvert.ycbcr2rgb,
                colorconvert.rgb2ycbcr,
        ):
            try:
                self.run_training(method, device="cpu")
            except Exception:
                print(f"---- FOR METHOD {method} ----")
                raise

    @unittest.skipIf(not torch.cuda.is_available(), "No CUDA available")
    def test_510_is_differentiable_cuda(self):
        for method in (
                colorconvert.hsv2rgb,
                colorconvert.rgb2hsv,
                colorconvert.lab2rgb,
                colorconvert.rgb2lab,
                colorconvert.rgb2xyz,
                colorconvert.xyz2rgb,
                colorconvert.xyz2lab,
                colorconvert.ycbcr2rgb,
                colorconvert.rgb2ycbcr,
        ):
            try:
                self.run_training(method, device="cuda")
            except Exception:
                print(f"---- FOR METHOD {method.__name__} ----")
                raise

    def run_training(self, method: Callable, batch_size: int = 64, steps: int = 1000, device: str = "auto"):
        devive = to_torch_device(device)

        data_batch = nn.Parameter(torch.rand(batch_size, 3, 8, 8).to(device))
        target_batch = torch.ones(batch_size, 3, 8, 8).to(device)

        optim = torch.optim.Adam([data_batch], lr=0.00001)

        losses = []
        for i in tqdm(range(steps), desc=f"method={method.__name__} device={device}"):
            output_batch = method(data_batch)

            loss = F.l1_loss(output_batch, target_batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss = float(loss)
            losses.append(loss)
            if math.isnan(loss):
                raise AssertionError(f"Encountered NaN loss: {losses}")

        self.assertLess(
            losses[-1],
            losses[0],
            f"No loss reduction, training failed!\nLosses: {losses}"
        )

