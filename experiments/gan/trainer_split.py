import math
import random
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torch
import torch.nn
import torch.fft
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util.image import signed_to_image, get_images_from_iterable
from src.train import Trainer
from src.util.module import num_module_parameters
from src.util.embedding import normalize_embedding


class TrainGANSplit(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self.model, "generator"), f"Module needs `generator` submodule"
        assert hasattr(self.model.generator, "num_inputs"), f"generator module needs `num_inputs` attribute"
        assert hasattr(self.model, "discriminator"), f"Module needs `discriminator` submodule"
        self._train_generator = False
        self._train_flip_count = 0
        self._discriminator_loss = 1.
        self._generator_loss = 1.
        self._train_generator_history = []
        self.logits_real = [1, 0]
        self.logits_fake = [0, 1]

        self.optimizers = [
            self.optimizers[0].__class__(self.model.generator.parameters(), **self.optimizers[0].defaults),
            self.optimizers[0].__class__(self.model.discriminator.parameters(), **self.optimizers[0].defaults),
        ]
        if self.schedulers:
            self.schedulers = [
                self.schedulers[0].__class__(self.optimizers[0], self.schedulers[0].T_max),
                self.schedulers[0].__class__(self.optimizers[1], self.schedulers[0].T_max),
            ]

        print(f"generator params:     {num_module_parameters(self.model.generator):,}")
        print(f"discriminator params: {num_module_parameters(self.model.discriminator):,}")

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        batch_size = input_batch.shape[0]

        if 0:
            self._train_flip_count += 1
            if ((self._train_generator and self._train_flip_count >= 20_000 // batch_size)
                    or (not self._train_generator and self._train_flip_count >= 2_000 // batch_size)):
                self._train_flip_count = 0
                self._train_generator = not self._train_generator

        else:
            _do_train_generator = self._generator_loss > self._discriminator_loss and self.num_input_steps > 20_000

            if _do_train_generator != self._train_generator:
                print(
                    "FLIPPING TRAINING TO", "generator" if _do_train_generator else "discriminator",
                    f", losses G {self._generator_loss:.4f}, D {self._discriminator_loss:.4f}"
                )
            self._train_generator = _do_train_generator

        do_train_generator = self._train_generator
        if random.random() < 0.01:
            do_train_generator = not do_train_generator

        if do_train_generator:
            self._skip_optimizer = [False, True]
        else:
            self._skip_optimizer = [True, False]

        self._train_generator_history.append(1 if do_train_generator else 0)
        if len(self._train_generator_history) >= 100:
            self.log_scalar(
                "generator_discriminator_train_ratio",
                sum(self._train_generator_history) / len(self._train_generator_history)
            )
            self._train_generator_history.clear()

        # -- generate adversarials --

        generator_input = self.create_generator_input_batch(batch_size)
        generated_batch = self.model.generator(generator_input)

        #if not do_train_generator:
        #    generated_batch = generated_batch.detach()

        #with torch.no_grad():
        #    random_ids = torch.randperm(batch_size)[:batch_size // 8].to(self.device)
        #    generated_batch[random_ids] = input_batch[random_ids]

        if input_batch.shape != generated_batch.shape:
            raise RuntimeError(
                f"Generator output shape mismatch"
                f", input_batch = {input_batch.shape}"
                f", generated_batch = {generated_batch.shape}"
                f", generator_input = {generator_input.shape}"
            )

        # -- run discriminator on real and fake data --

        discriminator_input_batch = torch.concat([
            self.transform_input_batch(input_batch),
            generated_batch,
        ])

        discriminator_output = self.model.discriminator(discriminator_input_batch)
        if discriminator_output.shape != torch.Size((batch_size * 2, 2)):
            raise RuntimeError(
                f"Discriminator output shape expected to be ({batch_size * 2}, 2), got {discriminator_output.shape}"
            )

        #if random.random() < .05:
        #    discriminator_output = discriminator_output[torch.randperm(discriminator_output.shape[0], device=self.device)]

        #if discriminator_output_real.shape != torch.Size((batch_size, 2)):
        #    raise RuntimeError(
        #        f"Discriminator output shape expected to be ({batch_size}, 2), got {discriminator_output_real.shape}"
        #    )
        #if discriminator_output_fake.shape != torch.Size((batch_size, 2)):
        #    raise RuntimeError(
        #        f"Discriminator output shape expected to be ({batch_size}, 2), got {discriminator_output_fake.shape}"
        #    )

        # -- setup target logits --

        logits_real = torch.Tensor([self.logits_real]).to(self.device).expand(batch_size, -1)

        logits_fake = torch.Tensor([self.logits_fake]).to(self.device).expand(batch_size, -1)

        target_logits = torch.concat([logits_real, logits_fake])

        # discriminator learns to predict real or fake
        discriminator_loss = self.loss_function(
            discriminator_output, target_logits + .01 * torch.randn_like(target_logits)
        )

        # DEBUGGING
        if not hasattr(self, "_print_output_count"):
            self._print_output_count = 0
        self._print_output_count += 1
        if self._print_output_count >= 1000:
            self._print_output_count = 0
            print(torch.concat([discriminator_output, target_logits], dim=-1).round(decimals=3))

        # generator learns to generate stuff that makes discriminator think it's real
        generator_loss = self.loss_function(
            discriminator_output[batch_size:], logits_real + .01 * torch.randn_like(logits_real)
        )
        #generator_loss = generator_loss + self.loss_function(
        #    discriminator_output[:batch_size], logits_fake + .01 * torch.randn_like(logits_real)
        #)

        # --- encourage generator diversity ---

        input_std = input_batch.std(dim=0).mean()
        generated_std = generated_batch.std(dim=0).mean()

        generator_std_loss = (input_std - generated_std).clamp_min(0.)

        acc_real = (discriminator_output[:batch_size].argmax(dim=-1) == 0).float().mean()
        acc_fake = (discriminator_output[batch_size:].argmax(dim=-1) == 1).float().mean()
        #print("XXX", discriminator_output_real.argmax(dim=-1))
        #print("YYY", discriminator_output_fake.argmax(dim=-1))

        # keep track of running losses
        self._discriminator_loss += .001 * (float(discriminator_loss) - self._discriminator_loss)
        self._generator_loss += .001 * (float(generator_loss)  - self._generator_loss)

        if do_train_generator:
            loss = generator_loss + 0.1 * generator_std_loss
        else:
            loss = discriminator_loss

        return {
            "loss": loss,
            "loss_discriminator": discriminator_loss,
            "loss_generator": generator_loss,
            "loss_generator_std": generator_std_loss,
            "accuracy_discriminator": (acc_real + acc_fake) / 2.,
            "accuracy_discriminator_real": acc_real,
            "accuracy_discriminator_fake": acc_fake,
        }

    def create_generator_input_batch(self, batch_size: int, seed: Optional[int] = None):
        if seed is not None:
            seed = torch.Generator().manual_seed(seed)

        batch = torch.randn(batch_size, self.model.generator.num_inputs, generator=seed).to(self.device)
        return normalize_embedding(batch)

    def write_step(self):
        batch_size = 64

        generated_batch = self.model.generator(
            self.create_generator_input_batch(batch_size, seed=23)
        )

        if generated_batch.ndim == 4:
            self.log_image("generated_images", make_grid(generated_batch.clamp(0, 1)))

        validation_batch = []
        count = 0
        for batch in self.iter_validation_batches():
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            validation_batch.append(batch)
            count += batch.shape[0]
            if count >= batch_size:
                break
        validation_batch = torch.concat(validation_batch)[:batch_size]
        assert validation_batch.shape[0] == batch_size

        if 0:
            discriminator_output_real = self.model.discriminator(validation_batch)
            discriminator_output_fake = self.model.discriminator(generated_batch)
            #print("REAL", discriminator_output_real)
            #print("FAKE", discriminator_output_fake)

            self.log_scalar(
                "discriminator_validation_accuracy_real",
                (discriminator_output_real.argmax(dim=-1) == 0).float().mean()
            )
            self.log_scalar(
                "discriminator_validation_accuracy_fake",
                (discriminator_output_fake.argmax(dim=-1) == 1).float().mean()
            )

        combined_input_batch = torch.concat([validation_batch, generated_batch])
        predicted_batch = self.model.discriminator(combined_input_batch)

        images = []
        logits_real = torch.Tensor([self.logits_real]).to(self.device).expand(predicted_batch.shape[0], -1)
        logits_fake = torch.Tensor([self.logits_fake]).to(self.device).expand(predicted_batch.shape[0], -1)

        diff = (logits_real - predicted_batch).abs().mean(-1)
        for i in range(2):
            for idx in diff.argsort(-1):
                images.append(combined_input_batch[idx])

            diff = (logits_fake - predicted_batch).abs().mean(-1)

        self.log_image(
            "discriminator_thinks_real_fake", make_grid(images, nrow=batch_size * 2)
        )
