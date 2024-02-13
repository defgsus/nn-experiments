import math
import unittest
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.clipig.source_models import *
from src.models.transform import Reshape


class SimpleAutoencoder(nn.Module):

    def __init__(self, shape: Tuple[int, int, int], code_size: int):
        super().__init__()

        self.shape = shape
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(math.prod(shape), code_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, math.prod(shape)),
            Reshape(shape),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TestSourceModel(unittest.TestCase):

    def test_pixelmodel(self):
        for size in (
                (32, 32),
                (320, 32),
                (32, 107),
        ):
            for model_class in (PixelModel, PixelHSVModel):

                exc_msg = f"size={size}, class={model_class.__name__}"

                model = model_class(
                    size=size,
                    channels="RGB",
                )

            model.clear()
            model.randomize()
            for image_shape in (
                    (3, 100, 100),
                    (3, 21, 320),
                    (1, 300, 31),
            ):
                model.set_image(torch.rand(*image_shape))

            with torch.no_grad():
                pixels = model()
                self.assertEqual(
                    (3, size[1], size[0]),
                    tuple(pixels.shape),
                    exc_msg
                )

    def test_autoencoder(self):
        for shape in (
                (3, 32, 32),
                (3, 24, 32),
                (3, 32, 48),
        ):
            for grid_size in (
                    (1, 1),
                    (2, 3),
                    (3, 2),
            ):
                code_size = 128

                exc_msg = f"shape={shape}, grid_size={grid_size}"

                model = AutoencoderModelHxW(
                    autoencoder=SimpleAutoencoder(shape, code_size),
                    autoencoder_shape=shape,
                    code_size=code_size,
                    grid_size=grid_size,
                    overlap=(0, 0),
                )

                model.clear()
                model.randomize()
                for image_shape in (
                        (3, 100, 100),
                        (3, 21, 320),
                        (1, 300, 31),
                ):
                    model.set_image(torch.rand(*image_shape))

                with torch.no_grad():
                    pixels = model()
                    self.assertEqual(
                        (3, shape[-2] * grid_size[-1], shape[-1] * grid_size[-2]),
                        tuple(pixels.shape),
                        exc_msg
                    )
