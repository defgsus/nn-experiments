import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.base import *
from src.datasets import DissimilarImageIterableDataset


class TestImageDissimilar(TestBase):

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
