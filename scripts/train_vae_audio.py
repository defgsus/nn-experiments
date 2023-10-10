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
from src.util.embedding import *
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


class VectorQuantization(nn.Module):

    def __init__(self, code_size: int, codebook_size, code_std: float = 0.01):
        super().__init__()
        self.code_size = code_size
        self.codebook_size = codebook_size
        self.codebook = nn.Parameter(
            code_std * torch.randn(self.codebook_size, self.code_size),
        )
        self.codebook_histogram = {}

    def forward(self, x):
        embeddings = normalize_embedding(x)
        similarities = embeddings @ self.codebook.T
        # return similarities

        best_indices = similarities.argmax(1)
        if self.training:
            #if self.codebook.grad is not None:
            #    print("X", self.codebook.grad.abs().sum())
            #with torch.no_grad():
            #print(best_indices.detach().tolist())
            for idx in best_indices:
                idx = int(idx)
                self.codebook_histogram[idx] = self.codebook_histogram.get(idx, 0) + 1
                # mix = 1. / self.codebook_histogram[idx]
                # self.codebook[idx] = self.codebook[idx] * (1. - mix) + mix * embeddings[idx]

        return self.codebook[best_indices]


class VQVAEAudioConv(VariationalAutoencoder):
    def __init__(
            self,
            slice_size: int,
            latent_dims: int,
            channels: Iterable[int],
            kernel_size: int = 5,
            codebook_size: int = 16,
            **kwargs,
    ):
        channels = list(channels)
        encoder = Conv1dBlock(
            channels=channels,
            kernel_size=kernel_size,
            act_fn=nn.ReLU(),
        )
        encoded_shape = encoder.get_output_shape((1, slice_size))
        encoder_dims = math.prod(encoded_shape)

        encoder = nn.Sequential(
            encoder,
            nn.Flatten(),
            # nn.Dropout(.5),
            nn.Linear(encoder_dims, 1024),
            VectorQuantization(1024, 1024),
        )
        encoder_dims = 1024

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

    def forward(self, x):
        y = super().forward(x)

        if False and self.training:
            with torch.no_grad():
                if self.encoder.encoder[0].layers[0].weight.grad is not None:
                    print("FIRST", self.encoder.encoder[0].layers[0].weight.grad.abs().sum())
                if self.encoder.encoder[-1].codebook.grad is not None:
                    print("CODEBOOK", self.encoder.encoder[-1].codebook.grad.abs().sum())
                    self.encoder.encoder[-2].grad = self.encoder.encoder[-1].codebook.grad
        return y


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

    def run_validation(self):
        vq = self.model.encoder.encoder[-1]
        if hasattr(vq, "codebook"):
            if hasattr(self, "_previous_codebook"):
                diff = (self._previous_codebook - vq.codebook).abs().mean()
                print("CODEBOOK change", float(diff))
            self._previous_codebook = vq.codebook.detach()[:]

            print({k: vq.codebook_histogram[k] for k in sorted(vq.codebook_histogram)})
            # print(vq.codebook.std(1))

        super().run_validation()


def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SLICE_SIZE = 1024 * 4
    LATENT_SIZE = 128
    SAMPLERATE = 44100

    ds = AudioSliceIterableDataset(
        "~/Music/", recursive=True,
        #"~/Music/COIL - Absinthe/COIL - animal are you.mp3",
        sample_rate=SAMPLERATE,
        slice_size=SLICE_SIZE,
        stride=50,
        interleave_files=200,
        shuffle_files=True,
        mono=True,
        #verbose=True,
    )
    #ds = TransformIterableDataset(ds, transforms=[Reshape((1, SLICE_SIZE))])
    ds = IterableShuffle(ds, max_shuffle=10_000)

    test_ds = AudioSliceIterableDataset(
        "~/Music/King Crimson/Vrooom Vrooom Disc 1/01 Vrooom Vrooom.mp3",
        sample_rate=SAMPLERATE,
        slice_size=SLICE_SIZE,
        stride=SLICE_SIZE // 4,
        interleave_files=20,
        mono=True,
    )
    test_ds = IterableShuffle(test_ds, max_shuffle=10_000)
    test_ds = LimitIterableDataset(test_ds, 10_000)
    #test_ds = TransformIterableDataset(test_ds, transforms=[Reshape((1, SLICE_SIZE))])

    if test_ds is None:
        train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))
    else:
        train_ds = ds

    sample = next(iter(ds))
    assert sample.shape == torch.Size((1, SLICE_SIZE)), sample.shape
    sample = next(iter(test_ds))
    assert sample.shape == torch.Size((1, SLICE_SIZE)), sample.shape

    #model = VAEAudioLinear(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, kl_loss_weight=10.)
    #model = VAEAudioConv(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, channels=[1, 16, 24, 32])
    model = VAEAudioConv(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, channels=[1, 16, 16, 16], kernel_size=15, kl_loss_weight=0.01)
    #model = VQVAEAudioConv(slice_size=SLICE_SIZE, latent_dims=LATENT_SIZE, channels=[1, 16, 16, 16])
    print(model)

    trainer = TrainAudioAutoencoder(
        **kwargs,
        model=model,
        #min_loss=0.001,
        #num_epochs_between_validations=1,
        num_inputs_between_validations=100_000 if isinstance(train_ds, IterableDataset) else None,
        num_inputs_between_checkpoints=1_000_000,
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
        if not trainer.load_checkpoint():
            trainer.load_checkpoint("best")

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
