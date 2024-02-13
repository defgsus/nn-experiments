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
