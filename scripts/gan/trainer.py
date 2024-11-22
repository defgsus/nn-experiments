import dataclasses
import json
import os
import re
import math
import random
import itertools
import argparse
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union, Generator, Dict

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util import *
from src.util.image import signed_to_image, get_images_from_iterable
from src.models.util import get_loss_callable
from src.models.cnn import *
from src.datasets import *


@dataclasses.dataclass
class GANTrainerSettings:

    # -- mandatory (though optional for scripts)
    path: Union[str, Path] = None
    generator: nn.Module = None
    discriminator: nn.Module = None
    train_dataset: Union[BaseDataset, BaseIterableDataset] = None
    validation_dataset: Union[BaseDataset, BaseIterableDataset] = None
    code_size: int = None

    # -- optional --
    # "minmax", "wasserstein", "class", "fake_class"
    loss_type: str = "minmax"
    num_classes: int = 0  # require extra class_logits from dataset

    num_epochs: int = 100
    batch_size: int = 64
    steps_per_module: int = 64
    steps_per_discriminator_ratio: float = 1.
    switch_if_loss_below: Union[None, bool, float] = None
    discriminator_bootstrap_steps: Optional[int] = None
    device: str = "auto"


@dataclasses.dataclass
class DiscriminatorOutput:
    critic: Optional[torch.Tensor] = None
    class_logits: Optional[torch.Tensor] = None


class GANTrainer:

    PROJECT_PATH = Path(__file__).resolve().parent.parent.parent

    def __init__(
            self,
            settings: GANTrainerSettings,
    ):
        self.checkpoint_path = self.PROJECT_PATH / "checkpoints" / settings.path
        self.tensorboard_path = self.PROJECT_PATH / "runs" / settings.path
        print(f"checkpoints: {self.checkpoint_path}")
        print(f"tensorboard: {self.tensorboard_path}")

        self.device = to_torch_device(settings.device)

        self.settings = settings
        self.generator = settings.generator.to(self.device)
        self.discriminator = settings.discriminator.to(self.device)
        self.train_dataset = settings.train_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=settings.batch_size)
        self.validation_dataset = settings.validation_dataset

        self.epoch = 0
        self.global_step = 0
        self.optimizer_generator = torch.optim.AdamW(self.generator.parameters(), lr=0.0001)
        self.optimizer_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0001)
        self._writer = None
        self._current_steps_per_module = 0
        self._train_discriminator = True
        self._discriminator_bootstrap_steps = self.settings.discriminator_bootstrap_steps or 0
        self._losses = {
            True: 10.,
            False: 10.,
        }
        self._loss_smooth = 0.01
        self._scalars: Dict[str, List[float]] = {}
        self._scalar_log_frames = 10

    def train(self):
        for self.epoch in range(self.settings.num_epochs):
            self.train_epoch()
            self.write_step()

    def train_epoch(self):
        try:
            total = len(self.train_dataset)
        except:
            total = None

        def _tqdm_desc():
            names = {
                True:  "discrimin",
                False: "generator",
            }
            return f"epoch #{self.epoch} {names[self._train_discriminator]}"

        with tqdm(
                total=total,
                desc=_tqdm_desc(),
        ) as progress:
            for batch_idx, batch in enumerate(self.train_dataloader):

                if isinstance(batch, (tuple, list)):
                    batch = [
                        i.to(self.device) if isinstance(i, torch.Tensor) else i
                        for i in batch
                    ]
                    batch_size = batch[0].shape[0]
                else:
                    batch = batch.to(self.device)
                    batch_size = batch.shape[0]

                if self._train_discriminator:
                    self.train_discriminator_batch(batch_size, batch)
                else:
                    self.train_generator_batch(batch_size)

                progress.update(batch_size)
                self.global_step += batch_size
                self._current_steps_per_module += batch_size

                # -- switch between discriminator/generator --

                do_switch = False
                max_steps = self.settings.steps_per_module
                if self._train_discriminator:
                    max_steps = int(max_steps * self.settings.steps_per_discriminator_ratio)
                if self._current_steps_per_module >= max_steps:
                    do_switch = True

                if self.settings.switch_if_loss_below is not None:
                    if self.settings.switch_if_loss_below is True:
                        if self._losses[self._train_discriminator] < self._losses[not self._train_discriminator]:
                            # print("switching at losses", self._losses)
                            do_switch = True
                    elif isinstance(self.settings.switch_if_loss_below, float):
                        if self._losses[self._train_discriminator] < self.settings.switch_if_loss_below:
                            do_switch = True

                if self._discriminator_bootstrap_steps > 0:
                    self._discriminator_bootstrap_steps -= batch_size
                    if self._discriminator_bootstrap_steps > 0:
                        do_switch = False

                if do_switch:
                    self._current_steps_per_module = 0
                    self._train_discriminator = not self._train_discriminator
                    progress.desc = _tqdm_desc()

                self.log_scalar("do_train_discriminator", float(self._train_discriminator))

    def train_discriminator_batch(self, batch_size: int, batch):
        self.discriminator.train()
        self.generator.eval()

        real_batch = batch
        real_logits = None
        if isinstance(real_batch, (list, tuple)):
            real_logits: torch.Tensor = real_batch[1]
            real_batch: torch.Tensor = real_batch[0]

            if self.settings.loss_type in ("class", "fake_class"):
                real_logits = torch.cat([
                    real_logits,
                    torch.zeros((real_logits.shape[0], 1)).to(real_logits)
                ], dim=-1)

        if self.settings.num_classes > 0:
            if real_logits is None:
                raise AssertionError(f"`num_discriminator_classes` is set but dataset provides no labels")

        fake_batch = self.generate_images(batch_size)

        output_real: DiscriminatorOutput = self.discriminator(real_batch)
        output_fake: DiscriminatorOutput = self.discriminator(fake_batch)

        if self.settings.loss_type == "wasserstein":
            assert output_real.critic is not None
            # compare with: https://developers.google.com/machine-learning/gan/loss
            loss = (
                (2. - (output_fake.critic - output_real.critic)).abs()
            ).mean()

        elif self.settings.loss_type == "minmax":
            assert output_real.critic is not None
            ones = torch.ones_like(output_real.critic)
            loss_real = F.l1_loss(output_real.critic, ones)
            loss_fake = F.l1_loss(output_fake.critic, -ones)
            loss = loss_real + loss_fake
            self.log_scalar("discriminator_batch_fake_min", output_fake.critic.min())
            self.log_scalar("discriminator_batch_fake_max", output_fake.critic.max())

        elif self.settings.loss_type == "fake_class":
            assert self.settings.num_classes > 0
            assert output_fake.class_logits is not None
            loss = F.cross_entropy(
                output_fake.class_logits,
                # expect [0, 0, ..., 1] (all fake)
                torch.Tensor([[0] * self.settings.num_classes + [1]]).to(output_fake.class_logits).repeat(batch_size, 1),
            )
            # self.log_scalar("train_loss_discriminator_fake_label", loss)

        elif self.settings.loss_type == "class":
            assert self.settings.num_classes > 0
            assert output_fake.class_logits is not None
            loss = F.cross_entropy(
                torch.concat([
                    output_real.class_logits,
                    output_fake.class_logits,
                ]),
                torch.concat([
                    real_logits,
                    # expect [0, 0, ..., 1] (all fake)
                    torch.Tensor([[0] * self.settings.num_classes + [1]]).to(output_fake.class_logits).repeat(batch_size, 1),
                ])
            )

        else:
            raise NotImplementedError(f"no loss_type '{self.settings.loss_type}'")

        if False: #self.settings.train_label:
            assert self.settings.num_classes > 0
            assert output_real.class_logits is not None

            if real_logits.shape != output_real.class_logits.shape:
                raise RuntimeError(
                    f"shapes dont match: {real_logits.shape} {output_real.class_logits.shape}"
                )

            label_loss = F.cross_entropy(output_real.class_logits, real_logits)
            loss = loss + label_loss
            self.log_scalar("train_loss_discriminator_label", label_loss)

        loss.backward()
        self.optimizer_discriminator.step()
        self.optimizer_discriminator.zero_grad()
        # clip_module_weights(self.discriminator, 1)

        self.log_scalar("train_loss_discriminator", loss)
        self._losses[True] += (float(loss) - self._losses[True]) * self._loss_smooth

    def train_generator_batch(self, batch_size: int):
        self.generator.train()
        self.discriminator.eval()

        fake_batch = self.generate_images(batch_size)
        output_fake = self.discriminator(fake_batch)

        if self.settings.loss_type == "wasserstein":
            assert output_fake.critic is not None
            loss = (1. - output_fake.critic).abs().mean()

        elif self.settings.loss_type == "minmax":
            assert output_fake.critic is not None
            ones = torch.ones_like(output_fake)
            loss = F.l1_loss(output_fake, ones)

        elif self.settings.loss_type in ("class", "fake_class"):
            assert self.settings.num_classes > 0
            assert output_fake.class_logits is not None

            loss = F.cross_entropy(
                # construct [is-predicted-label, or-fake-label]
                torch.cat([
                    output_fake.class_logits[:, :-1].max(dim=-1, keepdim=True)[0],
                    output_fake.class_logits[:, -1:],
                ], dim=-1),
                # and push it to [1, 0]
                torch.Tensor([[1, 0]]).to(output_fake.class_logits).repeat(batch_size, 1),
            )

        else:
            raise NotImplementedError(f"no loss_type '{self.settings.loss_type}'")

        loss.backward()
        self.optimizer_generator.step()
        self.optimizer_generator.zero_grad()

        self.log_scalar("train_loss_generator", loss)
        self._losses[False] += (float(loss) - self._losses[False]) * self._loss_smooth

    def generate_images(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        code_batch = torch.randn((batch_size, self.settings.code_size), generator=generator).to(self.device)
        return self.generator(code_batch)

    def write_step(self):
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            fake_batch = self.generate_images(batch_size=8 * 8, seed=23)
            self.log_image("images_generated", make_grid(fake_batch, nrow=8))

            output_fake: DiscriminatorOutput = self.discriminator(fake_batch)

            if output_fake.class_logits is not None:
                self._log_classify_grid(
                    "images_generated_classified",
                    fake_batch, output_fake.class_logits,
                )

                real_batch = next(iter(DataLoader(self.validation_dataset, batch_size=8 * 8)))
                if isinstance(real_batch, (list, tuple)):
                    real_logits: torch.Tensor = real_batch[1].to(self.device)
                    real_batch: torch.Tensor = real_batch[0].to(self.device)

                    output_real: DiscriminatorOutput = self.discriminator(real_batch)

                    #self._log_classify_grid(
                    #    "images_real_classified",
                    #    real_batch, output_real.class_logits,
                    #)

                    self._log_classify_grid(
                        "images_mixed_classified",
                        torch.concat([real_batch, fake_batch]),
                        torch.concat([output_real.class_logits, output_fake.class_logits]),
                    )

    def _log_classify_grid(self, name: str, images: torch.Tensor, class_logits: torch.Tensor):
        num_classes = class_logits.shape[-1]
        empty_image = torch.zeros_like(images[0])
        grid = [
            #[empty_image for _ in range(num_classes)]
            [empty_image] * num_classes
            for _ in range(images.shape[0])
        ]
        max_indices = class_logits.argmax(dim=-1)
        for i, row in enumerate(grid):
            row[max_indices[i]] = images[i]

        grid = sum(grid, [])
        self.log_image(name, make_grid(grid, nrow=num_classes))

    def log_scalar(self, tag: str, value):
        if isinstance(value, torch.Tensor):
            value = float(value.detach())
        if tag not in self._scalars:
            self._scalars[tag] = []
        self._scalars[tag].append(value)
        if len(self._scalars[tag]) > self._scalar_log_frames:
            value = sum(self._scalars[tag]) / len(self._scalars[tag])
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.global_step)
            self._scalars[tag].clear()

    def log_image(self, tag: str, image: torch.Tensor):
        try:
            image = image.clamp(0, 1)
            self.writer.add_image(tag=tag, img_tensor=image, global_step=self.global_step)
        except TypeError as e:
            warnings.warn(f"logging image `{tag}` failed, {type(e).__name__}: {e}")

    @property
    def writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(str(self.tensorboard_path))
        return self._writer

