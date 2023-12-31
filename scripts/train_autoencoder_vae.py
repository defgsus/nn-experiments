import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union

import torchvision.models
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
from clip.model import VisionTransformer

from src import console
from src.train.train_autoencoder import TrainAutoencoder
from src.models.cnn import *
from src.models.generative import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters
from src.algo import Space2d
from src.models.vae import *
from src.models.transform import *

from scripts.train_classifier_dataset import AlexNet
from scripts import datasets


class ConvEncoder(nn.Module):

    def __init__(self, channels: Iterable[int], shape: Tuple[int, int, int], code_size: int = 100):
        super().__init__()
        self.channels = list(channels)
        self.shape = shape
        encoder_block = Conv2dBlock(channels=self.channels, kernel_size=5, act_fn=nn.GELU())
        conv_shape = (self.channels[-1], *encoder_block.get_output_shape(self.shape[-2:]))
        self.layers = torch.nn.Sequential(
            encoder_block,
            nn.Flatten(),
            nn.Linear(math.prod(conv_shape), code_size),
        )

    def forward(self, x):
        return self.layers(x)


class MLPEncoder(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int],
            final_activation: Callable = nn.Sigmoid(),
    ):
        super().__init__()
        self.channels = [math.prod(shape), *channels]
        self.shape = shape
        self.layers = torch.nn.Sequential()
        self.final_activation = final_activation

        for i, (chan, next_chan) in enumerate(zip(self.channels, self.channels[1:])):
            self.layers.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.decoder.append(nn.ReLU())

    def forward(self, x):
        x = x.flatten(1)
        z = self.layers(x)
        return self.final_activation(z)

    def weight_images(self, **kwargs):
        images = []
        for w in self.layers[-1].weight.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        return images


class MLPDecoder(nn.Module):

    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int],
            final_activation: Callable = nn.Sigmoid(),
    ):
        super().__init__()
        self.channels = [*channels, math.prod(shape)]
        self.shape = shape
        self.layers = torch.nn.Sequential()
        self.final_activation = final_activation

        for i, (chan, next_chan) in enumerate(zip(self.channels, self.channels[1:])):
            self.layers.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.decoder.append(nn.ReLU())

    def forward(self, z):
        x = self.layers(z).reshape(-1, *self.shape)
        return self.final_activation(x)

    def weight_images(self, **kwargs):
        images = []
        for w in self.layers[-1].weight.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        return images


class VariationalAutoencoderAlexMLP(VariationalAutoencoder):
    def __init__(self, shape: Tuple[int, int, int], encoder_dims: int, latent_dims: int):
        encoder = VariationalEncoder(
            encoder=AlexNet(num_classes=encoder_dims),
            encoder_dims=encoder_dims,
            latent_dims=latent_dims,
        )
        decoder = MLPDecoder(shape, [latent_dims])
        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )
        self.shape = shape


class VariationalAutoencoderMLP(VariationalAutoencoder):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            latent_dims: int,
            hidden_dims: Iterable[int] = tuple(),
    ):
        channels = [*hidden_dims, latent_dims]
        encoder = MLPEncoder(
            shape=shape,
            channels=channels,
        )
        encoder = VariationalEncoder(
            encoder=encoder,
            encoder_dims=latent_dims,
            latent_dims=latent_dims,
        )

        channels = list(reversed(channels))
        decoder = MLPDecoder(
            shape=shape,
            channels=channels,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            #reconstruction_loss_weight=10.,
            #kl_loss_weight=10.,
        )


class VariationalAutoencoderConv(VariationalAutoencoder):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            latent_dims: int,
            kernel_size: int = 15,
            channels: Iterable[int] = (16, 32),
            encoder_dims: Optional[int] = None,
    ):
        channels = [shape[0], *channels]
        encoder = Conv2dBlock(
            channels=channels,
            kernel_size=kernel_size,
            act_fn=nn.ReLU(),
        )
        encoded_shape = encoder.get_output_shape(shape)
        encoder = nn.Sequential(
            encoder,
            nn.Flatten(),
            nn.Dropout(.5),
        )
        if encoder_dims is None:
            encoder_dims = math.prod(encoded_shape)
        else:
            nn.Linear(math.prod(encoded_shape), encoder_dims)

        encoder = VariationalEncoder(
            encoder=encoder,
            encoder_dims=encoder_dims,
            latent_dims=latent_dims,
        )

        channels = list(reversed(channels))
        decoder = Conv2dBlock(
            channels=channels,
            kernel_size=kernel_size,
            act_fn=nn.ReLU(),
            transpose=True,
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dims, math.prod(encoded_shape)),
            Reshape(encoded_shape),
            decoder,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )



def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    test_ds = None
    if 0:
        SHAPE = (3, 64, 64)
        #ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{SHAPE[-2]}x{SHAPE[-1]}.pt"))
        ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{128}x{128}.pt"))
        ds = TransformDataset(
            ds,
            dtype=torch.float, multiply=1. / 255.,
            transforms=[
                VT.CenterCrop(64),
                #VT.RandomCrop(SHAPE[-2:])
            ],
            num_repeat=1,
        )
    elif 0:
        SHAPE = (1, 32, 32)

        ds = datasets.all_patch_dataset(SHAPE)

        test_ds = make_image_patch_dataset(
            path="~/Pictures/diffusion", recursive=True,
            shape=SHAPE,
            scales=[1./12., 1./6, 1./3, 1.], stride=SHAPE[-2:],
            interleave_images=10, image_shuffle=10,
            #transforms=[lambda x: VF.resize(x, tuple(s // 6 for s in x.shape[-2:]))], stride=5,
            max_size=2000,
        )
        assert next(iter(ds))[0].shape[-3:] == torch.Size(SHAPE), next(iter(ds))[0].shape

    else:
        SHAPE = (3, 32, 32)

        ds = RpgTileIterableDataset()
        test_ds = SplitIterableDataset(ds, ratio=20, train=False)
        ds = SplitIterableDataset(ds, ratio=20, train=True)

    if test_ds is None:
        train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))
    else:
        train_ds = ds

    # model = VariationalAutoencoderAlexMLP(SHAPE, 512, 128)  # not good in reproduction
    model = VariationalAutoencoderConv(SHAPE, channels=[16, 24, 32], kernel_size=5, latent_dims=128)
    #model = VariationalAutoencoderMLP(SHAPE, 10)
    print(model)

    trainer = TrainAutoencoder(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        num_inputs_between_validations=100_000 if isinstance(train_ds, IterableDataset) else None,
        data_loader=DataLoader(train_ds, batch_size=256, shuffle=not isinstance(train_ds, IterableDataset)),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        training_noise=.2,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=0.000001),
            #torch.optim.Adadelta(model.parameters(), lr=.1),
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
