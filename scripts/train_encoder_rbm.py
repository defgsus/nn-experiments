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
from src.models.rbm import *
from src.models.generative import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters
from src.algo import Space2d
from src.models.encoder import Encoder2d
from src.models.transform import *


class BoltzmanEncoder(Encoder2d):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            hidden: Iterable[int] = (),
            dropout: float = 0.,
    ):
        self._hidden = list(hidden)
        self._dropout = dropout
        code_sizes = [code_size]
        if hidden is not None:
            code_sizes += self._hidden
        super().__init__(shape, code_size)
        self.rbms = nn.Sequential()
        for code_size, next_code_size in zip([math.prod(shape)] + code_sizes, code_sizes):
            self.rbms.append(
                RBM(code_size, next_code_size, dropout=dropout)
            )

    @property
    def device(self):
        return self.rbms[0].weight.device

    def forward(self, x):
        for rbm in self.rbms:
            x = rbm(x)
        return x

    def train_step(self, input_batch) -> torch.Tensor:
        data = input_batch
        if isinstance(data, (tuple, list)):
            data = data[0]
        loss = None
        for rbm in self.rbms:
            loss_ = rbm.train_step(data)
            loss = loss_ if loss is None else loss + loss_
            data = rbm.forward(data)
        return loss

    def weight_images(self, **kwargs):
        return self.rbms[0].weight_images(**kwargs)

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "hidden": self._hidden,
            "dropout": self._dropout,
        }
    @classmethod
    def from_data(cls, data: dict):
        extra = data["_extra_state"]
        model = cls(
            shape=extra["shape"],
            code_size=extra["code_size"],
            hidden=extra.get("hidden") or [],
            dropout=extra.get("dropout") or 0.,
        )
        if "rbm.weight" in data:
            data = {
                "_extra_state": data["_extra_state"],
                "rbms.0.weight": data["rbm.weight"],
                "rbms.0.bias_visible": data["rbm.bias_visible"],
                "rbms.0.bias_hidden": data["rbm.bias_hidden"],
            }
        model.load_state_dict(data)
        return model


class TrainerEncoder(Trainer):
    def write_step(self):
        shape = self.hparams["shape"]
        images = []
        count = 0
        for batch in self.iter_validation_batches():
            images.append(batch[0])
            count += batch[0].shape[0]
            if count >= 32:
                break
        images = torch.cat(images)[:32].to(self.device)

        outputs = self.model.forward(images)
        self.log_image("validation_features", outputs.unsqueeze(0))

        org, recon = self.model.contrastive_divergence(images[:8])
        recon2 = self.model.gibbs_sample(images[:8], num_steps=20)
        self.log_image("reconstruction", make_grid(
            get_images_from_iterable(org.view(8, *shape), squeezed=True, num=8)
            + get_images_from_iterable(recon.view(8, *shape), squeezed=True, num=8)
            + get_images_from_iterable(recon2.view(8, *shape), squeezed=True, num=8),
            nrow=8
        ))


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
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
    else:
        SHAPE = (1, 32, 32)
        def _stride(shape: Tuple[int, int]):
            # print(shape)
            size = min(shape)
            if size <= 512:
                return 5
            return SHAPE[-2:]

        ds = make_image_patch_dataset(
            path="~/Pictures/photos", recursive=True,
            shape=SHAPE,
            scales=[1./12., 1./6, 1./3, 1.], stride=_stride,
            interleave_images=20, image_shuffle=30,
            # transforms=[lambda x: VF.resize(x, tuple(s // 6 for s in x.shape[-2:]))], stride=5,
        )
        test_ds = make_image_patch_dataset(
            path="~/Pictures/diffusion", recursive=True,
            shape=SHAPE,
            scales=[1./12., 1./6, 1./3, 1.], stride=_stride,
            interleave_images=10, image_shuffle=10,
            #transforms=[lambda x: VF.resize(x, tuple(s // 6 for s in x.shape[-2:]))], stride=5,
            max_size=2000,
        )
        assert next(iter(ds))[0].shape[-3:] == torch.Size(SHAPE), next(iter(ds))[0].shape

    if test_ds is None:
        train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))
    else:
        train_ds = ds

    #model = VariationalAutoencoderConv(SHAPE, channels=[16, 24, 32], kernel_size=5, latent_dims=128)
    model = BoltzmanEncoder(SHAPE, 128, [1024])
    print(model)

    trainer = Trainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        num_inputs_between_validations=1_000_000 if isinstance(train_ds, IterableDataset) else None,
        data_loader=DataLoader(train_ds, batch_size=1024, shuffle=not isinstance(train_ds, IterableDataset)),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        training_noise=.2,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.0001),# weight_decay=0.000001),
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
