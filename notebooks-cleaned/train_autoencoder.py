import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader, TensorDataset

from src import console
from src.train.train_autoencoder import TrainAutoencoder
from src.models.cnn import Conv2dBlock
from src.datasets import ImageFolderIterableDataset, ImageAugmentation, IterableShuffle


class LinearStack(torch.nn.Module):

    def __init__(
            self,
            sizes: List[int],
            act_fn: Optional[torch.nn.Module] = torch.nn.Tanh(),
            act_last_layer: bool = False,
            bias: bool = True,
            batch_norm: bool = True
    ):
        super().__init__()
        self.sizes = list(sizes)
        self.layers = torch.nn.Sequential()
        if batch_norm:
            self.layers.append(torch.nn.BatchNorm1d(self.sizes[0]))
        for i, (size, next_size) in enumerate(zip(self.sizes, self.sizes[1:])):
            self.layers.append(torch.nn.Linear(size, next_size, bias=bias))
            if act_last_layer or i < len(self.sizes) - 2:
                self.layers.append(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)


class SimpleAutoEncoder(torch.nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int = 128,
            act_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
    ):
        super().__init__()
        self.shape = shape
        self.code_size = code_size
        self._shape_flat = math.prod(self.shape)
        self.act_fn = act_fn

        sizes = [self._shape_flat, self.code_size * 2, self.code_size]
        self.encoder = LinearStack(sizes, act_fn=self.act_fn, act_last_layer=True)
        self.decoder = LinearStack(list(reversed(sizes)), act_fn=self.act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encode(x)
        return self.decode(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self._shape_flat)
        x = self.encoder.forward(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder.forward(x).reshape(-1, *self.shape)
        return x

    def weight_images(self):
        images = []
        for layers in (self.encoder.layers, self.decoder.layers):
            for layer in layers:
                if hasattr(layer, "weight"):
                    weight = layer.weight
                    if weight.ndim == 2:
                        images.append(weight)
        return images


class ConvAutoEncoder(torch.nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int = 128,
            max_channels: int = 100,
            act_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = code_size
        self.act_fn = act_fn

        assert self.shape[-2] == self.shape[-1], self.shape

        kernel_size = 7
        #channels = list(map(int, torch.linspace(shape[0], max_channels, 4)))
        channels = [shape[0], 32, 32]
        encoder_block = Conv2dBlock(channels=channels, kernel_size=kernel_size, act_fn=self.act_fn)#, act_last_layer=True)
        conv_shape = (channels[-1], *encoder_block.get_output_shape(self.shape[-2:]))
        self.encoder = torch.nn.Sequential(
            encoder_block,
            nn.Flatten(),
            nn.Linear(math.prod(conv_shape), code_size),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(code_size, math.prod(conv_shape)),
            nn.Unflatten(1, conv_shape),
            Conv2dBlock(channels=list(reversed(channels)), kernel_size=kernel_size, act_fn=self.act_fn, transpose=True),#, act_last_layer=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encode(x)
        return self.decode(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder.forward(x).reshape(-1, *self.shape)
        return x

    def XX_weight_images(self):
        images = []
        for layers in (self.encoder.layers, self.decoder.layers):
            for layer in layers:
                if hasattr(layer, "weight"):
                    weight = layer.weight
                    if weight.ndim == 2:
                        images.append(weight)
        return images


def create_augmentation():
    return torch.nn.Sequential(
        VT.RandomAffine(degrees=20, scale=(.1, 4), translate=(.5, .5)),
        VT.RandomPerspective(p=.5, distortion_scale=.7),
        VT.RandomInvert(p=.3),
        VT.RandomVerticalFlip(),
        VT.RandomHorizontalFlip(),
    )

def augmented_dataset(shape: Tuple[int, int, int], ds, num_aug=1):
    return ImageAugmentation(
        ds,
        augmentations=[
            create_augmentation()
            for i in range(num_aug)
        ],
        final_shape=shape[-2:],
        final_channels=shape[0],
    )


def create_dataset(shape: Tuple[int, int, int]):
    ds = ImageFolderIterableDataset(
        root=Path("~/Pictures/__diverse/").expanduser(),
        #root=Path("~/Pictures/eisenach/").expanduser(),
    )
    ds = augmented_dataset(shape, ds, num_aug=4)
    #ds = IterableShuffle(ds, max_shuffle=5)
    return ds


def main():
    SHAPE = (3, 32, 32)

    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    # ds = TensorDataset(torch.load("./datasets/diverse-64x64-aug4.pt"))
    ds = TensorDataset(torch.load("./datasets/diverse-32x32-aug16.pt"))
    assert ds[0][0].shape[:3] == torch.Size(SHAPE), ds[0][0].shape

    #model = SimpleAutoEncoder(shape=SHAPE)
    model = ConvAutoEncoder(shape=SHAPE)
    print(model)

    trainer = TrainAutoencoder(
        **kwargs,
        model=model,
        data_loader=DataLoader(ds, shuffle=True, batch_size=10),
        validation_loader=DataLoader(
            augmented_dataset(SHAPE, ImageFolderIterableDataset(Path("~/Pictures/eisenach/").expanduser()))
        ),
        optimizers=[
            #torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001),
            torch.optim.Adadelta(model.parameters(), lr=0.1),
        ]
    )

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
    #block = Conv2dBlock([3, 1, 1, 1, 1, 1, 1, 1, 1], kernel_size=5)
    #print(block.get_output_shape((64, 64)))
