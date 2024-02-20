import tempfile
from pathlib import Path
import unittest
from typing import Iterable, List

import torch
import torch.nn as nn

from src.models.encoder import EncoderConv2d
from src.tests.base import TestBase


class TestEncoderConv(TestBase):

    def test_100_save_load(self):
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)

            enc1 = EncoderConv2d(shape=(3, 100, 90), kernel_size=21, code_size=96, act_fn=nn.LeakyReLU())
            torch.save(enc1.state_dict(), path / "model.pt")

            enc2 = EncoderConv2d.from_torch(path / "model.pt")

            self.assertNotEqual(id(enc1), id(enc2))
            self.assertNotEqual(id(enc1.convolution.layers[0].weight), id(enc2.convolution.layers[0].weight))

            self.assertTensorEqual(enc1.convolution.layers[0].weight, enc2.convolution.layers[0].weight)
