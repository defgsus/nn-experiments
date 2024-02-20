import time
import unittest
import math
from itertools import permutations
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as VF

from src.models.transform import FFTLayer
from src.tests.base import TestBase


class TestFFTLayer(TestBase):

    def test_100_all(self):
        for type, allow_complex, concat_dim, shape, expected_shape in (
                ("fft", False, -1, (1, 24, 32), (1, 24, 64)),
                ("fft", True,  -1, (1, 24, 32), (1, 24, 32)),
                ("fft", False, -1, (1, 3, 24, 32), (1, 3, 24, 64)),
                ("fft", False, -2, (1, 3, 24, 32), (1, 3, 48, 32)),
                ("fft", False, -3, (1, 3, 24, 32), (1, 6, 24, 32)),
                ("fft", False, -4, (1, 3, 24, 32), (2, 3, 24, 32)),

                ("rfft", False, -1, (1, 24, 32), (1, 24, 34)),
                ("rfft", False, -2, (1, 24, 32), (1, 48, 17)),
                ("rfft", False, -3, (1, 24, 32), (2, 24, 17)),
                ("rfft", False, -4, (1, 1, 24, 32), (2, 1, 24, 17)),

                ("hfft", False, -1, (1, 24, 32), (1, 24, 62)),
                ("hfft", False, -2, (1, 24, 32), (1, 24, 62)),
                ("hfft", True,  -1, (1, 24, 32), (1, 24, 62)),

        ):
            for norm in ("forward", "backward", "ortho"):

                error_msg = f"for type={type}, cmplx={allow_complex}, dim={concat_dim}, norm={norm}"

                layer = FFTLayer(type=type, allow_complex=allow_complex, concat_dim=concat_dim, norm=norm)

                image = torch.rand(shape)
                output = layer(image)

                self.assertEqual(output.shape, torch.Size(expected_shape), error_msg)

                if not allow_complex:
                    self.assertFalse(torch.is_complex(output), error_msg)

                inverse_layer = FFTLayer(
                    type=type, allow_complex=allow_complex, concat_dim=concat_dim, norm=norm, inverse=True,
                )

                image_repro = inverse_layer(output)

                self.assertEqual(image.shape, image_repro.shape, error_msg)

                if type != "hfft":
                    self.assertTensorEqual(image, image_repro, error_msg, places=2)
