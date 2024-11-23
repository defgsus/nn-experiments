import math
import random
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict, Type

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
from src.train.train_autoencoder import TrainAutoencoder
from src.models.transform import Sobel
from .sampler import DiffusionSamplerBase, DiffusionSamplerNoise, DiffusionSamplerBlur


class DiffusionModelInput:

    def __init__(
            self,
            images: torch.Tensor,        # [B,C,H,W]
            *parameters: torch.Tensor,   # can be [B,C] or [B,C,H,W]
    ):
        assert images.ndim == 4, f"Expected 4 dimensions, got {images.shape}"

        for param in parameters:
            assert param.ndim in (2, 4), f"Expected 2 or 4 dimensions, got {param.shape}"
            assert param.shape[0] == images.shape[0], \
                f"Expected batch size {images.shape[0]}, got {param.shape}"

        self.images = images
        self.parameters = parameters

    def parameter_embedding(self):
        embeddings = []
        for param in self.parameters:
            if param.ndim == 2:
                param = param[:, :, None, None].repeat(1, 1, *self.images.shape[-2:])
            embeddings.append(param)
        return torch.cat(embeddings, dim=-3)


class DiffusionModelOutput:

    def __init__(
            self,
            noise: torch.Tensor,                           # [B,C,H,W]
    ):
        assert noise.ndim == 4, f"Expected 4 dimensions, got {noise.shape}"

        self.noise = noise


class TrainDiffusion(TrainAutoencoder):

    def __init__(
            self,
            generator_shape: Tuple[int, int, int],
            noise_scale: float = 1.,
            noise_amount_exponent: float = 1.,
            diffusion_sampler: DiffusionSamplerBase = DiffusionSamplerNoise(),
            num_class_logits: int = 0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generator_shape = generator_shape
        self.noise_scale = noise_scale
        self.noise_amount_exponent = noise_amount_exponent
        self.diffusion_sampler = diffusion_sampler
        self.num_class_logits = num_class_logits

    def add_noise(self, images: torch.Tensor, seed: Optional[int] = None):
        noise_amounts = self.diffusion_sampler.create_noise_amounts(
            batch_size=images.shape[0], seed=seed
        ).to(self.device)
        if self.noise_amount_exponent != 1.:
            noise_amounts = torch.pow(noise_amounts, self.noise_amount_exponent)
        return self.diffusion_sampler.add_noise(images, noise_amounts, seed=seed)

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        class_logits = None
        if isinstance(input_batch, (tuple, list)):
            if self.num_class_logits > 0:
                class_logits = input_batch[1]
            input_batch = input_batch[0]

        inputs = self.transform_input_batch(input_batch) * 2 - 1

        noisy_inputs, noise_amounts = self.add_noise(inputs)
        parameters = [noise_amounts]
        if self.num_class_logits > 0:
            parameters.append(class_logits)

        output = self.model(DiffusionModelInput(noisy_inputs, *parameters))
        assert isinstance(output, DiffusionModelOutput)

        outputs = noisy_inputs - output.noise

        reconstruction_loss = self.loss_function(outputs, inputs)

        return {
            "loss": reconstruction_loss,
        }

    def write_step(self):
        with torch.no_grad():
            for size, shape in (
                    (8, self.generator_shape),
                    (4, (self.generator_shape[-3], self.generator_shape[-2] * 2, self.generator_shape[-1] * 2)),
            ):
                image_list = []
                for steps in [1, 10, 20, 100, 200]:
                    print(f"generate images {shape}, steps={steps}")
                    images = self.generate_images(size * size, shape, seed=23, steps=steps)
                    image_list.append(make_grid(images, nrow=size))
                self.log_image(f"images_generated_{shape[-2]}x{shape[-1]}", make_grid(image_list, nrow=5))

        for batch in self.iter_validation_batches():
            class_logits = None
            if isinstance(batch, (tuple, list)):
                if self.num_class_logits > 0:
                    class_logits = batch[1][:4 * 8].to(self.device)
                batch = batch[0].to(self.device)

            batch = batch[:4 * 8]

            inputs = self.transform_input_batch(batch) * 2 - 1

            noisy_inputs, noise_amounts = self.add_noise(inputs, seed=42)
            parameters = [noise_amounts]
            if self.num_class_logits > 0:
                parameters.append(class_logits)

            output = self.model(DiffusionModelInput(noisy_inputs, *parameters))

            grid = []
            for image, noisy_image, noise in zip(inputs, noisy_inputs, output.noise):
                grid.append(image * .5 + .5)
                grid.append(noisy_image * .5 + .5)
                grid.append((noisy_image - noise) * .5 + .5)
                grid.append(noise * .5 + .5)

            self.log_image(f"images_validation", make_grid(grid, nrow=4 * 4).clamp(0, 1))
            break

    def generate_images(
            self,
            batch_size,
            shape: Tuple[int, int, int],
            steps: int = 10,
            seed: Optional[int] = None,
    ):
        if seed is not None:
            seed = torch.Generator().manual_seed(seed)

        noisy_images, noise_amounts = self.add_noise(
            torch.randn((batch_size, *shape), generator=seed).to(self.device),
            seed=seed,
        )
        if self.num_class_logits > 0:
            class_logits = torch.zeros((batch_size, self.num_class_logits)).to(self.device)
            for row in class_logits:
                row[torch.randint(0, self.num_class_logits, (1, ), generator=seed).item()] = 1

        outputs = noisy_images

        def _predict_noise(noisy_images: torch.Tensor, noise_amounts: torch.Tensor) -> torch.Tensor:
            parameters = [noise_amounts]
            if self.num_class_logits > 0:
                parameters.append(class_logits)
            return self.model(DiffusionModelInput(noisy_images, *parameters)).noise

        for step in range(steps):
            estimated_noise = _predict_noise(noisy_images, noise_amounts)
            outputs = self.diffusion_sampler.remove_noise(
                noisy_images,
                noise=estimated_noise,
                noise_amounts=noise_amounts / steps,
            )

            if step < steps - 1:
                noisy_images, _ = self.diffusion_sampler.add_noise(
                    outputs,
                    noise_amounts=noise_amounts / steps,
                    seed=seed,
                )

        return outputs.clamp(-1, 1) * .5 + .5

