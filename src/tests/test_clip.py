import torch
from torch.utils.data import DataLoader
import torchvision.transforms as VT

from tests.base import *
from src.models.clip import ClipSingleton


class TestClip(TestBase):

    def test_100_singleton(self):
        model1, pre1 = ClipSingleton.get(device="cpu")
        model2, pre2 = ClipSingleton.get(device="cpu")
        self.assertEqual(id(model1), id(model2))
        self.assertEqual(id(pre1), id(pre2))

    @unittest.skipIf(not torch.cuda.is_available(), "no cuda available")
    def test_110_singleton_per_device(self):
        model1, pre1 = ClipSingleton.get(device="cpu")
        model2, pre2 = ClipSingleton.get(device="cuda")
        self.assertNotEqual(id(model1), id(model2))
        self.assertNotEqual(id(pre1), id(pre2))

    def test_200_encode(self):
        # 1x2x3 -> 1x512
        self.assertEqual(
            (1, 512),
            ClipSingleton.encode_image(torch.Tensor(
                [
                    [[.1, .2, .3], [.4, .5, .6]]
                ],
            ), device="cpu").shape
        )
        # 1x1x2x3 -> 1x512
        self.assertEqual(
            (1, 512),
            ClipSingleton.encode_image(torch.Tensor(
                [
                    [
                        [[.1, .2, .3], [.4, .5, .6]]
                    ],
                ]
            ), device="cpu").shape
        )
        # 2x1x2x3 -> 1x512
        self.assertEqual(
            (2, 512),
            ClipSingleton.encode_image(torch.Tensor(
                [
                    [
                        [[.1, .2, .3], [.4, .5, .6]]
                    ],
                    [
                        [[.6, .5, .4], [.3, .2, .1]]
                    ],
                ]
            ), device="cpu").shape
        )
