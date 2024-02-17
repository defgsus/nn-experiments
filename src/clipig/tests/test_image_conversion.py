import unittest

import torch

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from src.clipig.app import util


class TestImageConversion(unittest.TestCase):

    def test_qimage_to_torch(self):
        image = QImage(QSize(100, 100), QImage.Format.Format_ARGB32)
        self.assertEqual(
            torch.Size((4, 100, 100)),
            util.qimage_to_torch(image).shape,
        )

    def test_torch_to_pil(self):
        for channels in (3, 4):
            height = 240
            width = 320

            msg = f"channels={channels}, width={width}, height={height}"

            image_torch = torch.Tensor(channels, height, width)
            image_pil = util.torch_to_pil(image_torch)

            self.assertEqual(
                (height, width),
                (image_pil.height, image_pil.width),
                msg
            )
