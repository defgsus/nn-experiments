import math
import argparse
from copy import deepcopy
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

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
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid
from clip.model import VisionTransformer

from src import console
from src.train import Trainer
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


class SpecToWave(nn.Module):

    def __init__(
            self,
            spec_shape: Tuple[int, int],
            out_size: int,
            n_bands: int,
    ):
        super().__init__()
        self.spec_shape = tuple(spec_shape)
        self.out_size = out_size
        self.n_bands = n_bands
        self.conv = Conv1dBlock((spec_shape[0], spec_shape[0], 30), kernel_size=5)
        out_shape = self.conv.get_output_shape(self.spec_shape)
        self.linear = nn.Linear(math.prod(out_shape), self.n_bands * 5)
        self._t = None

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        if spec.ndim == 4:
            if spec.shape[1] != 1:
                raise ValueError(f"Invalid spec shape {spec.shape}")
            spec = spec.squeeze(1)

        assert spec.shape[-2:] == self.spec_shape, spec.shape

        bs = spec.shape[0]
        c = self.conv(spec)
        b = self.linear(c.flatten(1)).view(-1, self.n_bands, 5)

        if self._t is None:
            self._t = torch.linspace(0, 1, self.out_size).to(c.dtype).to(c.device).view(1, 1, -1).expand(-1, self.n_bands, -1)
        t = self._t.expand(bs, -1, -1)

        freq = b[..., 0].unsqueeze(-1) * 100
        phase = b[..., 1].unsqueeze(-1)
        amp = b[..., 2].unsqueeze(-1)

        amp_freq = b[..., 3].unsqueeze(-1) * 1
        amp_phase = b[..., 4].unsqueeze(-1)
        amp = amp * torch.sin((t * amp_freq + amp_phase) * 6.28)

        wave = torch.sin((t * freq + phase) * 6.28) * amp
        return wave.mean(-2).unsqueeze(1)


class SpecToAudioTrainer(Trainer):

    def __init__(self, speccer: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.speccer = deepcopy(speccer).to(self.device)

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        audio_batch, spec_batch = input_batch

        recon_audio_batch = self.model(spec_batch)
        recon_spec_batch = self.speccer(recon_audio_batch)[..., :spec_batch.shape[-1]]
        return F.mse_loss(recon_spec_batch, spec_batch)
        #return F.mse_loss(recon_audio_batch, audio_batch)

    def write_step(self):
        for audio_batch, spec_batch in self.iter_validation_batches():
            break
        spec_batch = spec_batch[:16].to(self.device)

        recon_audio_batch = self.model(spec_batch)
        recon_spec_batch = self.speccer(recon_audio_batch)[..., :spec_batch.shape[-1]]

        self.log_image("validation_reconstruction", make_grid(
            [VF.vflip(s) for s in spec_batch]
            + [VF.vflip(s) for s in recon_spec_batch]
            , nrow=spec_batch.shape[0], normalize=True
        ))


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SAMPLE_RATE = 44_100
    SPEC_SHAPE = (64, 64)
    AUDIO_SIZE = SAMPLE_RATE // 4

    train_ds = AudioSpecIterableDataset(
        "~/Music", recursive=True,
        slice_size=SAMPLE_RATE * 4,
        stride=SAMPLE_RATE * 2,
        spec_shape=SPEC_SHAPE,
        spec_slice_size=SAMPLE_RATE // 4,
        spec_stride=1,
        interleave_files=1000,
        mono=True,
    )
    speccer = train_ds.speccer
    train_ds = TransformIterableDataset(train_ds, transforms=[lambda a: a[..., -AUDIO_SIZE:]])
    test_ds = AudioSpecIterableDataset(
        datasets.AUDIO_FILENAMES_1,
        slice_size=SAMPLE_RATE * 4,
        stride=SAMPLE_RATE * 2,
        spec_shape=SPEC_SHAPE,
        spec_slice_size=SAMPLE_RATE // 4,
        spec_stride=SAMPLE_RATE * 4,
        interleave_files=1000,
        seek_offset=30,
        mono=True,
    )
    test_ds = TransformIterableDataset(test_ds, transforms=[lambda a: a[..., -AUDIO_SIZE:]])
    test_ds = LimitIterableDataset(test_ds, 1000)

    sample = next(iter(train_ds))
    assert sample[0].shape == (1, AUDIO_SIZE), sample[0].shape
    assert sample[1].shape == (1, *SPEC_SHAPE), sample[1].shape

    train_ds = IterableShuffle(train_ds, 100_000)

    if 1:
        model = SpecToWave(SPEC_SHAPE, AUDIO_SIZE, 10)
    else:
        model = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(math.prod(SPEC_SHAPE), math.prod(SPEC_SHAPE)),
            nn.ReLU(),
            nn.Linear(math.prod(SPEC_SHAPE), AUDIO_SIZE),
            nn.Tanh(),
            Reshape((1, AUDIO_SIZE)),
        )
        with torch.no_grad():
            model[-3].weight *= .1
    print(model)

    trainer = SpecToAudioTrainer(
        **kwargs,
        speccer=speccer,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        num_inputs_between_validations=100_000 if isinstance(train_ds, IterableDataset) else None,
        data_loader=DataLoader(train_ds, batch_size=128, num_workers=0, shuffle=not isinstance(train_ds, IterableDataset)),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        training_noise=.05,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=0.000001),
            #torch.optim.Adadelta(model.parameters(), lr=.1),
        ],
        hparams={
            "spec_shape": SPEC_SHAPE,
        },
        weight_image_kwargs={
            "spec_shape": SPEC_SHAPE,
        }
    )

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
