import torch
from torch.utils.data import DataLoader
import torchvision.transforms as VT

from tests.base import *
from src.models.cnn import ConvAutoEncoder


class TestGrowingConvAE(TestBase):

    def test_100_grow(self):
        code_size = 128
        init_shape = (3, 8, 8)
        batch_shape = torch.Size((1,) + init_shape)
        model = ConvAutoEncoder(shape=init_shape, channels=[10], code_size=code_size)
        print(model)
        self.assertEqual(torch.Size((1, code_size)), model.encode(torch.zeros(1, *init_shape)).shape)
        self.assertEqual(batch_shape, model.forward(torch.zeros(1, *init_shape)).shape)

        for init_shape in (
                (3, 12, 12),
                (3, 16, 16),
                (3, 20, 20),
                (3, 24, 24),
        ):
            batch_shape = torch.Size((1,) + init_shape)
            model.shape = init_shape
            model.add_layer(channels=10)
            print(model)

            self.assertEqual(torch.Size((1, code_size)), model.encode(torch.zeros(1, *init_shape)).shape)
            self.assertEqual(batch_shape, model.forward(torch.zeros(1, *init_shape)).shape)
