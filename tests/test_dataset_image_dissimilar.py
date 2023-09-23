import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as VF

from tests.base import *
from src.datasets import *


class TestDatasetImageDissimilar(TestBase):

    def test_100_batch_size(self):
        image_ds = TensorDataset(torch.rand(1000, 1, 32, 20), torch.linspace(0, 999, 1000).unsqueeze(1))

        ds = DissimilarImageIterableDataset(image_ds, max_similarity=.77, verbose=False)
        images, ids = list(DataLoader(ds, batch_size=1000))[0]
        self.assertLess(images.shape[0], 1000)

        for batch_size in (1, 2, 3, 10, 99, 10_000):
            ds = DissimilarImageIterableDataset(
                image_ds, max_similarity=.77, verbose=False, batch_size=batch_size
            )
            images2, ids2 = list(DataLoader(ds, batch_size=1000))[0]
            self.assertTensorEqual(images, images2)
            self.assertTensorEqual(ids, ids2)

    def test_1000_data_passing(self):
        image_ds = ImageFolderIterableDataset(self.DATA_PATH / "images", force_channels=1, with_filename=True)
        image_ds = TransformIterableDataset(image_ds, transforms=[lambda x: VF.crop(x, 0, 0, 32, 32)])
        image_ds = ImagePatchIterableDataset(image_ds, (15, 15), with_pos=True)
        num_images = len(list(image_ds))

        ds = DissimilarImageIterableDataset(
            image_ds, max_similarity=.8, verbose=False,
        )
        images, positions, filenames = list(DataLoader(ds, batch_size=1000))[0]
        self.assertLess(images.shape[0], num_images)
        self.assertEqual(
            (str(self.DATA_PATH / 'images/camera.jpg'),
             str(self.DATA_PATH / 'images/max.jpeg'),
             str(self.DATA_PATH / 'images/max.jpeg')),
            filenames,
        )
        self.assertTensorEqual(
            [[ 0,  0],
             [ 0,  0],
             [15,  0]],
            positions,
        )


