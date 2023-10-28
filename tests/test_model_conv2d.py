import unittest

import torch
import torch.nn.functional as F

from .base import TestBase
from src.models.cnn import *


class TestModelConv2d(TestBase):

    def test_300_reversing(self):
        for org_settings in (
            dict(
                channels=[3, 1],
                kernel_size=5,
            ),
            dict(
                channels=[3, 6, 9],
                kernel_size=5,
            ),
            dict(
                channels=[3, 4, 5],
                kernel_size=[7, 8],
            ),
            dict(
                channels=[3, 5, 7, 8],
                kernel_size=[5, 5, 6],
                space_to_depth=True,
            ),
        ):
            for extra_settings in (
                    dict(act_last_layer=True),
                    dict(act_last_layer=False),
            ):
                settings = {**org_settings, **extra_settings}

                conv_forward = Conv2dBlock(**settings)

                conv_backward = Conv2dBlock(
                    **{
                        **settings,
                        "transpose": True,
                        "channels": list(reversed(settings["channels"])),
                        "kernel_size": settings["kernel_size"] if isinstance(settings["kernel_size"], int)
                            else list(reversed(settings["kernel_size"])),
                    }
                )
                for from_transpose_call in (False, True):

                    image = torch.randn(1, settings["channels"][0], 64, 64)
                    image_recon = conv_backward(conv_forward(image))

                    self.assertEqual(
                        image.shape, image_recon.shape,
                        f"with settings: {settings}, create_transposed={from_transpose_call}",
                    )
                    conv_backward = conv_forward.create_transposed()
