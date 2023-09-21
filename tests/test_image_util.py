import torch
from torch.utils.data import DataLoader
import torchvision.transforms as VT

from tests.base import *
from src.util.image import *


class TestImageUtil(TestBase):

    def test_100_set_image_channels_to_1(self):
        # 1xHxW > 1xHxW
        self.assertTensorEqual(
            [
                [[1, 0], [0, 1], [.5, 0]],
            ],
            set_image_channels(torch.Tensor(
                [
                    [[1, 0], [0, 1], [.5, 0]],
                ]
            ), 1)
        )
        # 2xHxW > 1xHxW
        self.assertTensorEqual(
            [
                [[1, 0], [0, 1], [.5, 0]],
            ],
            set_image_channels(torch.Tensor(
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],  # second channel is discarded
                ]
            ), 1)
        )
        # 3xHxW > 1xHxW
        self.assertTensorEqual(
            [
                [[1.4729000329971313, 0.34200000762939453], [0.04560000076889992, 1.8149000406265259], [1.0299500226974487, 0.06840000301599503]],
            ],
            set_image_channels(torch.Tensor(
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],
                    [[0, 3], [.4, 3], [0, .6]],
                ]
            ), 1)
        )

    def test_101_set_image_channels_to_3(self):
        # 1xHxW > 3xHxW
        self.assertTensorEqual(
            [
                [[1, 0], [0, 1], [.5, 0]],
                [[1, 0], [0, 1], [.5, 0]],
                [[1, 0], [0, 1], [.5, 0]],
            ],
            set_image_channels(torch.Tensor(
                [
                    [[1, 0], [0, 1], [.5, 0]],
                ]
            ), 3)
        )
        # 2xHxW > 3xHxW
        self.assertTensorEqual(
            [
                [[1, 0], [0, 1], [.5, 0]],
                [[1, 0], [0, 1], [.5, 0]],
                [[1, 0], [0, 1], [.5, 0]],
            ],
            set_image_channels(torch.Tensor(
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],  # second channel is discarded
                ]
            ), 3)
        )
        # 3xHxW > 3xHxW
        self.assertTensorEqual(
            [
                [[1, 0], [0, 1], [.5, 0]],
                [[2, 0], [0, 2], [1.5, 0]],
                [[0, 3], [.4, 3], [0, .6]],
            ],
            set_image_channels(torch.Tensor(
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],
                    [[0, 3], [.4, 3], [0, .6]],
                ]
            ), 3)
        )

    def test_110_set_image_channels_to_1_batched(self):
        # 1xHxW > 1xHxW
        self.assertTensorEqual([
                [
                    [[1, 0], [0, 1], [.5, 0]],
                ],
                [
                    [[2, 0], [0, 2], [1.5, 0]],
                ],
            ],
            set_image_channels(torch.Tensor([
                [
                    [[1, 0], [0, 1], [.5, 0]],
                ],
                [
                    [[2, 0], [0, 2], [1.5, 0]],
                ]
            ]), 1)
        )
        # 2xHxW > 1xHxW
        self.assertTensorEqual([
            [
                [[1, 0], [0, 1], [.5, 0]],
            ],
            [
                [[2, 0], [0, 2], [1.5, 0]],
            ],
        ],
            set_image_channels(torch.Tensor([
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],  # second channel is discarded
                ],
                [
                    [[2, 0], [0, 2], [1.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],  # second channel is discarded
                ]
            ]), 1)
        )
        # 3xHxW > 1xHxW
        self.assertTensorEqual(
            [
                [
                    [[1.4729000329971313, 0.34200000762939453], [0.04560000076889992, 1.8149000406265259], [1.0299500226974487, 0.06840000301599503]],
                ],
                [
                    [[1.1848000288009644, 0.34200000762939453], [0.04560000076889992, 1.5268000364303589], [0.7418500185012817, 0.06840000301599503]],
                ],
            ],
            set_image_channels(torch.Tensor([
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],
                    [[0, 3], [.4, 3], [0, .6]],
                ],
                [
                    [[2, 0], [0, 2], [1.5, 0]],
                    [[1, 0], [0, 1], [.5, 0]],
                    [[0, 3], [.4, 3], [0, .6]],
                ]
            ]), 1)
        )

    def test_111_set_image_channels_to_3_batched(self):
        # 1xHxW > 3xHxW
        self.assertTensorEqual(
            [
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[1, 0], [0, 1], [.5, 0]],
                    [[1, 0], [0, 1], [.5, 0]],
                ],
                [
                    [[0, 1], [2, 3], [.4, .5]],
                    [[0, 1], [2, 3], [.4, .5]],
                    [[0, 1], [2, 3], [.4, .5]],
                ],
            ],
            set_image_channels(torch.Tensor(
                [
                    [
                        [[1, 0], [0, 1], [.5, 0]],
                    ],
                    [
                        [[0, 1], [2, 3], [.4, .5]],
                    ],
                ]
            ), 3)
        )
        # 2xHxW > 3xHxW
        self.assertTensorEqual(
            [
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[1, 0], [0, 1], [.5, 0]],
                    [[1, 0], [0, 1], [.5, 0]],
                ],
                [
                    [[1, 2], [3, 4], [5, 6]],
                    [[1, 2], [3, 4], [5, 6]],
                    [[1, 2], [3, 4], [5, 6]],
                ],
            ],
            set_image_channels(torch.Tensor(
                [
                    [
                        [[1, 0], [0, 1], [.5, 0]],
                        [[2, 0], [0, 2], [1.5, 0]],  # second channel is discarded
                    ],
                    [
                        [[1, 2], [3, 4], [5, 6]],
                        [[2.1, 0.1], [0.1, 2.1], [1.6, .1]],  # second channel is discarded
                    ],
                ]
            ), 3)
        )
        # 3xHxW > 3xHxW
        self.assertTensorEqual(
            [
                [
                    [[1, 0], [0, 1], [.5, 0]],
                    [[2, 0], [0, 2], [1.5, 0]],
                    [[0, 3], [.4, 3], [0, .6]],
                ],
                [
                    [[1, 0], [0, 3], [.5, 0]],
                    [[2, 2], [0, 2], [1.5, 0]],
                    [[0, 3], [.4, 3], [0.4, .6]],
                ],
            ],
            set_image_channels(torch.Tensor(
                [
                    [
                        [[1, 0], [0, 1], [.5, 0]],
                        [[2, 0], [0, 2], [1.5, 0]],
                        [[0, 3], [.4, 3], [0, .6]],
                    ],
                    [
                        [[1, 0], [0, 3], [.5, 0]],
                        [[2, 2], [0, 2], [1.5, 0]],
                        [[0, 3], [.4, 3], [0.4, .6]],
                    ],
                ]
            ), 3)
        )
