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
from src.algo import AudioUnderstander

from scripts.train_classifier_dataset import AlexNet
from scripts import datasets


class SimpleVAE(VariationalAutoencoder):
    def __init__(
            self,
            shape: Tuple[int, ...],
            latent_dims: int,
            **kwargs,
    ):
        encoder = VariationalEncoder(
            encoder=nn.Flatten(1),
            encoder_dims=math.prod(shape),
            latent_dims=latent_dims,
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dims, math.prod(shape)),
            Reshape(shape),
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            **kwargs,
        )



def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SPEC_SHAPE = (128, 128)

    # train autoencoder on spec patches
    if 0:
        SHAPE = (1, 128, 8)
        STRIDE = (128, 1)
        PATCHES_PER_SLICE = math.prod(s1 // s2 for s1, s2 in zip(SPEC_SHAPE, STRIDE[-2:]))
        TOTAL_SLICES = 20103
        TOTAL_PATCHES = TOTAL_SLICES * PATCHES_PER_SLICE
        print(PATCHES_PER_SLICE, "patches per slice")

        train_ds = datasets.audio_slice_dataset(
            path="~/Music/", recursive=True,
            interleave_files=200,
            mono=True,
            spectral_shape=SPEC_SHAPE,
            spectral_patch_shape=SHAPE[-2:],
            spectral_patch_stride=STRIDE,
            #shuffle_slices=10_000,
            shuffle_files=True,
            spectral_normalize=1_000,
        )
        test_ds = datasets.audio_slice_dataset(
            path=datasets.AUDIO_FILENAMES_2,
            interleave_files=200,
            mono=True,
            spectral_shape=SPEC_SHAPE,
            spectral_patch_shape=SHAPE[-2:],
            seek_offset=30.,
            #shuffle_slices=10_000,
            spectral_normalize=1_000,
        )
        test_ds = LimitIterableDataset(test_ds, 10000)

        sample = next(iter(train_ds))
        assert sample.shape == SHAPE, sample.shape

        model = SimpleVAE(SHAPE, latent_dims=math.prod(SHAPE) // 8, kl_loss_weight=0)
        print(model)

    # train autoencoder on bag-of-words
    else:
        train_ds = datasets.audio_slice_dataset(
            path="~/Music/", recursive=True,
            interleave_files=200,
            mono=True,
            spectral_shape=SPEC_SHAPE,
            shuffle_files=True,
            spectral_normalize=1_000,
        )
        audio_encoder = AudioUnderstander.load("./models/au/au-1sec-3x256.pt")
        train_ds = TransformIterableDataset(
            train_ds,
            transforms=[lambda x: audio_encoder.encode_spectrum(x)]
        )

        test_ds = datasets.audio_slice_dataset(
            path=datasets.AUDIO_FILENAMES_2,
            interleave_files=200,
            mono=True,
            spectral_shape=SPEC_SHAPE,
            seek_offset=30.,
            #shuffle_slices=10_000,
            spectral_normalize=1_000,
        )
        test_ds = LimitIterableDataset(test_ds, 1000)
        test_ds = TransformIterableDataset(
            test_ds,
            transforms=[lambda x: audio_encoder.encode_spectrum(x)]
        )

        SHAPE = (1, 256 * 3)
        sample = next(iter(train_ds))
        assert sample.shape == SHAPE, sample.shape

        model = SimpleVAE(SHAPE, latent_dims=math.prod(SHAPE) // 8, kl_loss_weight=0)
        print(model)

    trainer = TrainAutoencoder(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        num_inputs_between_validations=1_000_000 if isinstance(train_ds, IterableDataset) else None,
        data_loader=DataLoader(train_ds, batch_size=1024, num_workers=2, shuffle=not isinstance(train_ds, IterableDataset)),
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
