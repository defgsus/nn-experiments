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
from src.train import Trainer
from src.models.cnn import *
from src.models.encoder import *
from src.datasets import *
from src.util.image import *
from src.util.audio import *
from src.util import num_module_parameters
from src.models.vae import *
from src.models.transform import *

from scripts import datasets


class AudioPredictor(nn.Module):
    def __init__(
            self,
            slice_size: int,
            num_input_slices: int,
            latent_dims: int,
            encoder: nn.Module,
            decoder: nn.Module,
            predictor: nn.Module,
            channels: int = 1,
    ):
        super().__init__()
        self.slice_size = slice_size
        self.num_input_slices = num_input_slices
        self.latent_dims = latent_dims
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.channels = channels

    def forward(self, audio_batch):
        assert audio_batch.ndim == 3
        assert audio_batch.shape[-1] == self.num_input_slices * self.slice_size, f"Got {audio_batch.shape}"

        batch_size = audio_batch.shape[0]
        slices = audio_batch.view(batch_size * self.num_input_slices, self.channels, self.slice_size)

        embeddings = self.encoder(slices)
        embeddings = embeddings.view(batch_size, self.channels, self.latent_dims * self.num_input_slices)

        future_embeddings = self.predictor(embeddings)
        return self.decoder(future_embeddings)

    def hallucinate(self, audio, num_steps: int = 1):
        assert audio.ndim == 2
        assert audio.shape[0] == self.channels
        num_audio_slices = audio.shape[1] // self.slice_size

        input_audio = audio[:, -(num_audio_slices * self.num_input_slices):]
        input_embeddings = self.encoder(input_audio.view(self.num_input_slices, self.channels, self.slice_size))
        for i in range(num_steps):
            pass


class TrainAudioPredictor(Trainer):

    def write_step(self):
        shape = (128, 128)

        inputs = []
        expected_outputs = []
        count = 0
        for inp, target in self.iter_validation_batches():
            inputs.append(inp)
            expected_outputs.append(target)
            count += inp.shape[0]
            if count >= 32:
                break
        inputs = torch.cat(inputs)[:32].to(self.device)
        expected_outputs = torch.cat(expected_outputs)[:32].to(self.device)

        output_batch = self.model.forward(inputs)
        grid_images = []
        for output, expected_output in zip(output_batch, expected_outputs):
            grid_images.append(plot_audio(
                [expected_output.squeeze(0), output.squeeze(0)], shape, tensor=True
            ))

        image = make_grid(grid_images, nrow=8)
        self.log_image("validation_prediction", image)


class FutureAudioSliceDataset(IterableDataset):
    def __init__(self, dataset, slice_size: int, num_input_slices: int):
        self.dataset = dataset
        self.slice_size = slice_size
        self.num_input_slices = num_input_slices

    def __iter__(self):
        for big_slice in self.dataset:
            if isinstance(big_slice, (tuple, list)):
                big_slice, rest = big_slice[0], big_slice[1:]
            else:
                rest = tuple()

            all_but_one = self.slice_size * self.num_input_slices
            yield (
                big_slice[..., :all_but_one],
                big_slice[..., all_but_one:],
                *rest,
            )


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SLICE_SIZE = 1024
    LATENT_SIZE = 128
    SAMPLERATE = 44100
    NUM_INPUT_SLICES = 8

    ds = AudioSliceIterableDataset(
        "~/Music/", recursive=True,
        #"~/Music/COIL - Absinthe/COIL - animal are you.mp3",
        sample_rate=SAMPLERATE,
        slice_size=SLICE_SIZE * (NUM_INPUT_SLICES + 1),
        stride=SLICE_SIZE // 4,
        interleave_files=100,
        shuffle_files=True,
        mono=True,
        #verbose=True,
    )
    ds = IterableShuffle(ds, max_shuffle=10_000)
    ds = FutureAudioSliceDataset(ds, slice_size=SLICE_SIZE, num_input_slices=NUM_INPUT_SLICES)

    test_ds = AudioSliceIterableDataset(
        "~/Music/King Crimson/Vrooom Vrooom Disc 1/01 Vrooom Vrooom.mp3",
        sample_rate=SAMPLERATE,
        slice_size=SLICE_SIZE * (NUM_INPUT_SLICES + 1),
        stride=SLICE_SIZE // 4,
        interleave_files=20,
        mono=True,
    )
    test_ds = IterableShuffle(test_ds, max_shuffle=10_000)
    test_ds = LimitIterableDataset(test_ds, 10_000)
    test_ds = FutureAudioSliceDataset(test_ds, slice_size=SLICE_SIZE, num_input_slices=NUM_INPUT_SLICES)

    if test_ds is None:
        train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))
    else:
        train_ds = ds

    sample = next(iter(ds))
    assert sample[0].shape == torch.Size((1, SLICE_SIZE * NUM_INPUT_SLICES)), sample[0].shape
    assert sample[1].shape == torch.Size((1, SLICE_SIZE)), sample[1].shape

    model = AudioPredictor(
        slice_size=SLICE_SIZE,
        latent_dims=LATENT_SIZE,
        num_input_slices=NUM_INPUT_SLICES,
        encoder=EncoderConv1d.from_torch(f"./models/encoder1d/conv-1x{SLICE_SIZE}-{LATENT_SIZE}.pt"),
        decoder=EncoderConv1d.from_torch(f"./models/encoder1d/conv-1x{SLICE_SIZE}-{LATENT_SIZE}-decoder.pt"),
        #predictor=nn.Linear(LATENT_SIZE * NUM_INPUT_SLICES, LATENT_SIZE),
        predictor=nn.Sequential(
            nn.Linear(LATENT_SIZE * NUM_INPUT_SLICES, LATENT_SIZE * NUM_INPUT_SLICES),
            nn.ReLU(),
            nn.Linear(LATENT_SIZE * NUM_INPUT_SLICES, LATENT_SIZE),
        )
    )
    print(model)

    trainer = TrainAudioPredictor(
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
        if not trainer.load_checkpoint():
            trainer.load_checkpoint("best")

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
