import torch
from torch.utils.data import DataLoader

from src.tests.base import *
from src.datasets import ImageFolderIterableDataset


class TestImageFolder(TestBase):

    def assert_data_loader(self, dl: DataLoader, with_subfolder: bool = False):
        IMAGE_SHAPES = [
            torch.Size([3, 330, 383]),
            torch.Size([3, 265, 250]),
            torch.Size([3, 274, 360]),
            torch.Size([3, 1062, 750]),
            torch.Size([3, 423, 561]),
            torch.Size([1, 600, 443]),
        ]
        if with_subfolder:
            IMAGE_SHAPES += [
                torch.Size([1, 239, 163]),
            ]

        images = list(dl)
        self.assertEqual(len(IMAGE_SHAPES), len(images))
        image_shapes = [i.shape[-3:] for i in images]
        for expected_shape, image in zip(IMAGE_SHAPES, images):
            self.assertIn(expected_shape, image_shapes)
            self.assertEqual(expected_shape, image.shape[-3:])

    def test_100_single_worker(self):
        ds = ImageFolderIterableDataset(self.DATA_PATH / "images")
        self.assert_data_loader(DataLoader(ds))

    def test_110_multi_worker(self):
        ds = ImageFolderIterableDataset(self.DATA_PATH / "images")
        for i in range(1, 8):
            self.assert_data_loader(DataLoader(ds, num_workers=i))

    def test_200_recursive(self):
        ds = ImageFolderIterableDataset(self.DATA_PATH / "images", recursive=True)
        self.assert_data_loader(DataLoader(ds), with_subfolder=True)
