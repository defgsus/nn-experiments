import unittest

import torch

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from src.clipig.app import util
from src.clipig.app.images import LImage, LImageTiling


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

    def test_limage_tiling(self):
        limage = LImage()
        limage.add_layer(image=QImage(QSize(40, 40), QImage.Format.Format_ARGB32))
        tiling = LImageTiling(tile_size=(10, 10))
        limage.ui_settings.project_random_tiling_map = True
        limage.ui_settings.tiling_map_size = (4, 4)
        limage.ui_settings_changed()
        limage.set_tiling(tiling)
        limage._update_tiles()
        self.assertEqual(4, len(limage._tile_map))
        self.assertEqual(4, len(limage._tile_map[0]))
        limage._tile_map = [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(0, 0), (0, 0), (0, 0), (0, 0)],
            [(3, 2), (2, 2), (1, 0), (0, 0)],
        ]
        self.assertEqual(
            [(0, 0), (5, 6), (10, 11), (5, 2)],
            limage.pixel_pos_to_image_pos(
                [(0, 0), (5, 6), (10, 11), (15, 22)]
            )
        )
        # no projection
        self.assertEqual(
            [(QRect(0, 0, 5, 6), (0, 0)), (QRect(10, 11, 6, 7), (0, 0))],
            limage.pixel_rects_to_image_rects([QRect(0, 0, 5, 6), QRect(10, 11, 6, 7)])
        )
        # x overlaps
        self.assertEqual(
            [(QRect(7, 0, 3, 7), (0, 0)), (QRect(0, 0, 2, 7), (3, 0))],
            limage.pixel_rects_to_image_rects([QRect(7, 20, 5, 7), ])
        )
        # y overlaps
        self.assertEqual(
            [(QRect(5, 15, 5, 5), (0, 0)), (QRect(5, 0, 5, 2), (0, 5))],
            limage.pixel_rects_to_image_rects([QRect(5, 15, 5, 7), ])
        )
        # the full map/image
        self.assertEqual(
            [(QRect(0, 0, 10, 10), (0, 0)), (QRect(10, 0, 10, 10), (10, 0)), (QRect(20, 0, 10, 10), (20, 0)), (QRect(30, 0, 10, 10), (30, 0)), 
             (QRect(0, 10, 10, 10), (0, 10)), (QRect(10, 10, 10, 10), (10, 10)), (QRect(20, 10, 10, 10), (20, 10)), (QRect(30, 10, 10, 10), (30, 10)), 
             (QRect(0, 0, 10, 10), (0, 20)), (QRect(0, 0, 10, 10), (10, 20)), (QRect(0, 0, 10, 10), (20, 20)), (QRect(0, 0, 10, 10), (30, 20)),
             (QRect(30, 20, 10, 10), (0, 30)), (QRect(20, 20, 10, 10), (10, 30)), (QRect(10, 0, 10, 10), (20, 30)), (QRect(0, 0, 10, 10), (30, 30))],
            limage.pixel_rects_to_image_rects([QRect(0, 0, 40, 40), ])
        )
