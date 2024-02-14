import unittest

import torch

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from src.clipig.app import util
from src.clipig.app.images import LImage


class TestLImage(unittest.TestCase):

    def test_limage_rects(self):
        limage = LImage()
        limage.add_layer(image=QImage(QSize(100, 101), QImage.Format.Format_ARGB32))

        self.assertEqual(
            QSize(100, 101),
            limage.size(),
        )

        self.assertEqual(
            QRect(0, 0, 100, 101),
            limage.rect(),
        )

        self.assertEqual(
            QRect(0, 0, 100, 101),
            limage.content_rect(),
        )
