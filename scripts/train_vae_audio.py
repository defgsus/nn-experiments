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
from src.util.audio import *
from src.util import num_module_parameters
from src.models.vae import *
from src.models.transform import *

from scripts.train_classifier_dataset import AlexNet
from scripts import datasets


class VAEAudioLinear(VariationalAutoencoder):
    def __init__(
            self,
            slice_size: int,
            latent_dims: int,
            channels: int = 1,
            **kwargs,
    ):
        encoder = nn.Linear(slice_size, latent_dims)
        encoder = VariationalEncoder(
            encoder=encoder,
            encoder_dims=latent_dims,
            latent_dims=latent_dims,
        )

        decoder = nn.Sequential(
            nn.Linear(latent_dims, slice_size),
            Reshape((channels, slice_size)),
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            **kwargs,
        )


class VAEAudioConv(VariationalAutoencoder):
    def __init__(
            self,
            slice_size: int,
            latent_dims: int,
            channels: Iterable[int],
            kernel_size: int = 5,
            **kwargs,
    ):
        channels = list(channels)
        encoder = Conv1dBlock(
            channels=channels,
            kernel_size=kernel_size,
            act_fn=nn.ReLU(),
        )
        encoded_shape = encoder.get_output_shape((1, slice_size))
        encoder = nn.Sequential(
            encoder,
            nn.Flatten(),
            # nn.Dropout(.5),
        )

        encoder_dims = math.prod(encoded_shape)
        #if encoder_dims is None:
        #    encoder_dims = math.prod(encoded_shape)
        #else:
        #    nn.Linear(math.prod(encoded_shape), encoder_dims)

        encoder = VariationalEncoder(
            encoder=encoder,
            encoder_dims=encoder_dims,
            latent_dims=latent_dims,
        )

        channels = list(reversed(channels))
        decoder = Conv1dBlock(
            channels=channels,
            kernel_size=kernel_size,
            act_fn=nn.ReLU(),
            transpose=True,
            act_last_layer=False,
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dims, math.prod(encoded_shape)),
            Reshape(encoded_shape),
            decoder,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            **kwargs,
        )


class TrainAudioAutoencoder(TrainAutoencoder):

    def write_step(self):
        shape = (128, 128)

        waves = []
        count = 0
        for batch in self.iter_validation_batches():
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            waves.append(batch)
            count += batch.shape[0]
            if count >= 32:
                break
        waves = torch.cat(waves)[:32].to(self.device)

        output_batch = self.model.forward(waves)
        grid_images = []
        for i in range(0, waves.shape[0], 8):
            for j in range(8):
                if i + j < waves.shape[0]:
                    grid_images.append(plot_audio(
                        [waves[i + j].squeeze(0), output_batch[i + j].squeeze(0)], shape, tensor=True
                    ))
            #for j in range(8):
            #    if i + j < waves.shape[0]:
            #        grid_images.append(plot_audio(output_batch[i + j], shape, tensor=True))

        image = make_grid(grid_images, nrow=8)
        self.log_image("validation_reconstruction", image)

        features = self.model.encoder(waves)
        self.log_image("validation_features", signed_to_image(features))


def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SLICE_SIZE = 1024
    LATENT_SIZE = 128
    SAMPLERATE = 44100

    ds = AudioSliceIterableDataset(
        "~/Music/", recursive=True,
        #"~/Music/COIL - Absinthe/COIL - animal are you.mp3",
        sample_rate=SAMPLERATE,
        slice_size=SLICE_SIZE,
        stride=SLICE_SIZE // 4,
        interleave_files=20,
        shuffle_files=True,
        mono=True,
        #verbose=True,
    )
    #ds = TransformIterableDataset(ds, transforms=[Reshape((1, SLICE_SIZE))])
    ds = IterableShuffle(ds, max_shuffle=100_000)

    test_ds = AudioSliceIterableDataset(
        "~/Music/King Crimson/Vrooom Vrooom Disc 1/01 Vrooom Vrooom.mp3",
        sample_rate=SAMPLERATE,
        slice_size=SLICE_SIZE,
        stride=SLICE_SIZE // 4,
        interleave_files=20,
        mono=True,
    )
    test_ds = IterableShuffle(test_ds, max_shuffle=100_000)
    test_ds = LimitIterableDataset(test_ds, 10_000)
    #test_ds = TransformIterableDataset(test_ds, transforms=[Reshape((1, SLICE_SIZE))])

    if test_ds is None:
        train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))
    else:
        train_ds = ds

    assert next(iter(ds)).shape == torch.Size((1, SLICE_SIZE)), next(iter(ds)).shape
    assert next(iter(test_ds)).shape == torch.Size((1, SLICE_SIZE)), next(iter(test_ds)).shape

    #model = VAEAudioLinear(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, kl_loss_weight=10.)
    #model = VAEAudioConv(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, channels=[1, 16, 24, 32])
    model = VAEAudioConv(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, channels=[1, 16, 16, 16], kernel_size=7)
    print(model)

    trainer = TrainAudioAutoencoder(
        **kwargs,
        model=model,
        #min_loss=0.001,
        #num_epochs_between_validations=1,
        num_inputs_between_validations=1_000_000 if isinstance(train_ds, IterableDataset) else None,
        data_loader=DataLoader(train_ds, batch_size=256, num_workers=4, shuffle=not isinstance(train_ds, IterableDataset)),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        training_noise=.2,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=0.000001),
            #torch.optim.Adadelta(model.parameters(), lr=.1),
        ],
        hparams={
            #"shape": SHAPE,
        },
        weight_image_kwargs={
            #"shape": SHAPE,
        }
    )

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
