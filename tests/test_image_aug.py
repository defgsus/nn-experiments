import torch
from torch.utils.data import DataLoader
import torchvision.transforms as VT

from tests.base import *
from src.datasets import ImageFolderIterableDataset, ImageAugmentation


class TestImageAugmentation(TestBase):

    def assert_data_loader(self, dl: DataLoader):
        IMAGE_SHAPES = [
            torch.Size([3, 330, 383]),
            torch.Size([3, 265, 250]),
            torch.Size([3, 274, 360]),
            torch.Size([3, 1062, 750]),
            torch.Size([3, 423, 561]),
            torch.Size([1, 600, 443]),
        ]
        images = list(dl)
        self.assertEqual(len(IMAGE_SHAPES), len(images))
        image_shapes = [i.shape[-3:] for i in images]
        for expected_shape, image in zip(IMAGE_SHAPES, images):
            self.assertIn(expected_shape, image_shapes)
            self.assertEqual(expected_shape, image.shape[-3:])

    def test_100_single_worker(self):
        ds = ImageFolderIterableDataset(self.DATA_PATH / "images")
        ds = ImageAugmentation(
            ds,
            augmentations=[
                VT.RandomRotation(20),
                VT.RandomPerspective(),
            ],
            final_shape=(1024, 1024)
        )
        images = list(DataLoader(ds))
        self.assertEqual(12, len(images))
        for i, img in enumerate(images):
            self.assertEqual(torch.Size([1024, 1024]), img.shape[-2:])

    def test_110_multi_worker(self):
        for num_workers in range(1, 8):
            # print("## workers:", num_workers, "#"*100)

            ds = ImageFolderIterableDataset(self.DATA_PATH / "images")
            ds = ImageAugmentation(
                ds,
                augmentations=[
                    VT.RandomRotation(20),
                    VT.RandomPerspective(),
                ],
                final_shape=(1024, 1024)
            )
            images = list(DataLoader(ds, num_workers=num_workers))

            self.assertEqual(12, len(images))
            for i, img in enumerate(images):
                self.assertEqual(torch.Size([1024, 1024]), img.shape[-2:])
