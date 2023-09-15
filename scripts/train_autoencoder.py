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


class Encoder(nn.Module):

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


class Sinus(nn.Module):
    def __init__(self, size: int, freq_scale: float = 3.):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(size) * freq_scale)
        self.phase = nn.Parameter(torch.randn(size) * 3.)

    def forward(self, x):
        return torch.sin(x * self.freq + self.phase)


class Decoder(FreescaleImageModule):

    def __init__(self, code_size: int = 100):
        super().__init__(num_in=code_size)
        self.layers = nn.Sequential(
            nn.Linear(code_size + 2, code_size),
            Sinus(code_size, 30),
            nn.Linear(code_size, code_size),
            Sinus(code_size, 20),
            nn.Linear(code_size, 3),
            nn.Linear(3, 3),

        )

    def forward_state(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        return self.layers(x)


class NewAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.encoder = Encoder([shape[0], 20, 20], shape=shape)
        self.decoder = Decoder()
        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x), shape=self.shape)


class TransformerAutoencoder(ConvAutoEncoder):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int] = None,
            kernel_size: int = 5,
            code_size: int = 128,
            patch_size: int = 32,
            act_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
            batch_norm: bool = False,
    ):
        assert shape[-2] == shape[-1], shape

        super(TransformerAutoencoder, self).__init__(
            shape=shape, channels=channels, kernel_size=kernel_size, code_size=code_size,
            act_fn=act_fn, batch_norm=batch_norm,
        )
        self.encoder = VisionTransformer(
            shape[-1], patch_size=patch_size, width=256, layers=10, heads=8, output_dim=code_size
        )

    def weight_images(self, **kwargs):
        pass


class AlexAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], code_size: int):
        from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
        super().__init__()
        self.shape = shape
        self.encoder = AlexNet(num_classes=code_size)
        #self.encoder = ResNet(
        #    block=Bottleneck,
        #    layers=[2, 2, 2, 2], num_classes=code_size,
        #)
        self.decoder = nn.Sequential(
            nn.Linear(code_size, code_size * 2),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(code_size * 2, math.prod(shape)),
        )
        self.decoder = Decoder(code_size)
        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x), shape=self.shape)#.reshape(-1, *self.shape)

    def weight_images(self, **kwargs):
        pass#return self.encoder.weight_images(**kwargs)


class MLPAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], channels: Iterable[int]):
        super().__init__()
        self.channels = [math.prod(shape), *channels]
        self.shape = shape
        self.code_size = self.channels[-1]
        self.encoder = nn.Sequential(
            nn.Flatten(),
        )
        for i, (chan, next_chan) in enumerate(zip(self.channels, self.channels[1:])):
            self.encoder.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.encoder.append(nn.ReLU())
                # self.encoder.append(nn.Dropout())

        self.decoder = nn.Sequential()
        channels = list(reversed(self.channels))
        for i, (chan, next_chan) in enumerate(zip(channels, channels[1:])):
            self.decoder.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.decoder.append(nn.ReLU())
                # self.decoder.append(nn.Dropout())

        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)).reshape(-1, *self.shape)

    def weight_images(self, **kwargs):
        images = []
        for w in self.encoder[1].weight.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        for w in self.decoder[-1].weight.T.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        return images


class MLPDetailAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], code_size: int):
        super().__init__()
        self.shape = shape
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(shape), code_size),
        )

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.coarse = nn.Linear(code_size, math.prod(shape))
                self.fine = nn.Sequential(
                    nn.Linear(math.prod(shape) + code_size, math.prod(shape))
                )
            def forward(self, x):
                y = self.coarse(x)
                y2 = self.fine(torch.cat([y, x], dim=-1))
                y = (y + y2).reshape(-1, *shape)
                return y

        self.decoder = Decoder()

        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)).reshape(-1, *self.shape)

    def weight_images(self, **kwargs):
        images = []
        for w in self.encoder[-1].weight.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        for w in self.decoder.coarse.weight.T.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        return images


class TrainAutoencoderSpecial(TrainAutoencoder):

    def train_step(self, input_batch) -> torch.Tensor:
        input_batch = input_batch[0]
        feature_batch = self.model.encode(input_batch)
        output_batch = self.model.decode(feature_batch)

        reconstruction_loss = self.loss_function(input_batch, output_batch)
        return reconstruction_loss

        active_ratio = (feature_batch.abs()).sum() / math.prod(feature_batch.shape)
        sparsity_loss = (1./100. - active_ratio).abs()
        #sparsity_loss = (0. - feature_batch.mean()).abs() + (1. - feature_batch.std()).abs()

        self.log_scalar("reconstruction_loss", reconstruction_loss)
        self.log_scalar("sparsity_loss", sparsity_loss)

        return reconstruction_loss + sparsity_loss


def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    if 1:
        SHAPE = (1, 64, 64)
        #ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{SHAPE[-2]}x{SHAPE[-1]}.pt"))
        ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{128}x{128}.pt"))
        ds = TransformDataset(
            ds,
            dtype=torch.float, multiply=1. / 255.,
            transforms=[
                VT.CenterCrop(64),
                #VT.RandomCrop(SHAPE[-2:])
                VT.Grayscale(),
            ],
            num_repeat=1,
        )
    else:
        SHAPE = (1, 32, 32)
        ds = TensorDataset(torch.load(f"./datasets/fonts-regular-{SHAPE[-2]}x{SHAPE[-1]}.pt"))
    assert ds[0][0].shape[:3] == torch.Size(SHAPE), ds[0][0].shape

    train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    #model = NewAutoEncoder(SHAPE)
    model = ConvAutoEncoder(SHAPE, channels=[8, 16, 24], kernel_size=7, code_size=128) # good one
    model = ConvAutoEncoder(SHAPE, channels=[8, 16, 24], kernel_size=7, code_size=128, bias=False, linear_bias=False, act_last_layer=False)
    #model = TransformerAutoencoder(SHAPE, channels=[8, 16, 24], kernel_size=7, code_size=64)
    #model = ConvAutoEncoder(SHAPE, channels=[32], kernel_size=7, code_size=64)
    #model = MLPAutoEncoder(SHAPE, [512])
    #model = MLPDetailAutoEncoder(SHAPE, 128)
    #model = VariationalAutoencoder(SHAPE, 128)
    print(model)

    trainer = TrainAutoencoderSpecial(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        data_loader=DataLoader(train_ds, batch_size=64, shuffle=True),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        loss_function="l2",
        optimizers=[
            #torch.optim.Adam(model.parameters(), lr=.001),#, weight_decay=0.00001),
            torch.optim.Adadelta(model.parameters(), lr=1.),
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
