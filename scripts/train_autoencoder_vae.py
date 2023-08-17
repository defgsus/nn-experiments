import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable

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

from scripts.train_classifier_dataset import AlexNet


# based on https://avandekleut.github.io/vae/
class VariationalEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_dims: int, latent_dims: int):
        super().__init__()
        self.encoder = encoder
        self.linear_mu = nn.Linear(encoder_dims, latent_dims)
        self.linear_sigma = nn.Linear(encoder_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0.

    def forward(self, x):
        # move sampler to GPU
        device = self.linear_mu.weight.device
        if self.N.loc.device != device:
            self.N.loc = self.N.loc.to(device)
            self.N.scale = self.N.scale.to(device)

        x = self.encoder(x)

        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def weight_images(self, **kwargs):
        images = []

        if isinstance(self.encoder, nn.Sequential):
            for layer in self.encoder:
                if hasattr(layer, "weight_images"):
                    images += layer.weight_images(**kwargs)
        else:
            if hasattr(self.encoder, "weight_images"):
                images += self.encoder.weight_images(**kwargs)

        return images


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder: VariationalEncoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def train_step(self, batch):
        x = batch[0]
        recon = self.forward(x)

        loss = ((x - recon) ** 2).sum() + self.encoder.kl
        return loss

    def weight_images(self, **kwargs):
        images = []
        if hasattr(self.encoder, "weight_images"):
            images += self.encoder.weight_images(**kwargs)
        if hasattr(self.decoder, "weight_images"):
            images += self.decoder.weight_images(**kwargs)

        return images


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


class MLPDecoder(nn.Module):

    def __init__(self, shape: Tuple[int, int, int], channels: Iterable[int]):
        super().__init__()
        self.channels = [*channels, math.prod(shape)]
        self.shape = shape
        self.layers = torch.nn.Sequential()

        for i, (chan, next_chan) in enumerate(zip(self.channels, self.channels[1:])):
            self.layers.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.decoder.append(nn.ReLU())

    def forward(self, x):
        return self.layers(x).reshape(-1, *self.shape)

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


class Reshape(nn.Module):
    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(-1, *self.shape)


class VariationalAutoencoderConv(VariationalAutoencoder):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            latent_dims: int,
            kernel_size: int = 15,
            channels: Iterable[int] = (16, 32),
    ):
        channels = [shape[0], *channels]
        encoder = Conv2dBlock(
            channels=channels,
            kernel_size=kernel_size,
            act_fn=nn.ReLU(),
        )
        encoded_shape = encoder.get_output_shape(shape)
        encoder_dims = 1024
        encoder = nn.Sequential(
            encoder,
            nn.Flatten(),
            nn.Linear(math.prod(encoded_shape), encoder_dims)
        )
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

    if 1:
        SHAPE = (3, 64, 64)
        #ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{SHAPE[-2]}x{SHAPE[-1]}.pt"))
        ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{128}x{128}.pt")[:5000])
        ds = TransformDataset(
            ds,
            dtype=torch.float, multiply=1. / 255.,
            transforms=[
                VT.CenterCrop(64),
                #VT.RandomCrop(SHAPE[-2:])
            ],
            num_repeat=40,
        )
    else:
        SHAPE = (1, 32, 32)
        ds = TensorDataset(torch.load(f"./datasets/fonts-regular-{SHAPE[-2]}x{SHAPE[-1]}.pt"))
    assert ds[0][0].shape[:3] == torch.Size(SHAPE), ds[0][0].shape

    train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    # model = VariationalAutoencoderAlexMLP(SHAPE, 512, 128)  # not good in reproduction
    model = VariationalAutoencoderConv(SHAPE, 128)
    print(model)

    trainer = TrainAutoencoder(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        data_loader=DataLoader(train_ds, batch_size=64, shuffle=True),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.0001),#, weight_decay=0.00001),
            #torch.optim.Adadelta(model.parameters(), lr=1.),
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
