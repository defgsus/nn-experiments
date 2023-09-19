import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable


from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src import console
from src.train.train_autoencoder import TrainAutoencoder
from src.models.cnn import *
from src.datasets import ImageFolderIterableDataset, ImageAugmentation, IterableShuffle, TotalCADataset
from src.util.image import get_images_from_iterable


class CropDataset(Dataset):
    def __init__(self, dataset: Dataset, shape: Tuple[int, int]):
        self.dataset = dataset
        self.shape = shape
        self.cropper = VT.RandomCrop(self.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        item = self.dataset[item]
        if isinstance(item, (list, tuple)):
            return [self._convert(self.cropper(item[0])), *item[1:]]
        return self._convert(self.cropper(item))

    def _convert(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.dtype != torch.float:
            image = image.to(torch.float)
        return image


def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (1, 32, 32)
    # ds = TensorDataset(torch.load("./datasets/diverse-64x64-aug4.pt"))
    #ds = TensorDataset(torch.load("./datasets/diverse-32x32-aug32.pt"))
    #ds = TensorDataset(torch.load("./datasets/diverse-32x32-std01.pt"))
    #ds = TensorDataset(torch.load("./datasets/fonts-regular-32x32.pt"))
    #ds = TensorDataset(torch.load("./datasets/photos-32x32-std01.pt"))
    #ds = TotalCADataset(SHAPE[-2:], num_iterations=10, init_prob=.5, wrap=True, transforms=[lambda x: x.unsqueeze(0)])
    ds = CropDataset(TensorDataset(torch.load("./datasets/ca-64x64-i10-p05.pt")), shape=SHAPE[-2:])
    assert ds[0][0].shape[:3] == torch.Size(SHAPE), ds[0][0].shape

    #ds = ConcatDataset([ds, ds2])

    train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    model = ConvAutoEncoder(SHAPE, channels=[32, 64], code_size=32)
    print(model)

    trainer = TrainAutoencoder(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_inputs_between_validations=100_000,
        #data_loader=DataLoader(train_ds, shuffle=True, batch_size=10),
        data_loader=DataLoader(train_ds, batch_size=50, num_workers=5),
        validation_loader=DataLoader(test_ds),
        optimizers=[
            #torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0.001),
            torch.optim.Adadelta(model.parameters(), lr=0.1),
        ],
        hparams={
            "shape": SHAPE,
        },
        weight_image_kwargs={
            "shape": SHAPE,
        }
    )

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
