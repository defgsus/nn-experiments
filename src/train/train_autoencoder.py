import math
import random
import itertools
import shutil
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
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
from .trainer import Trainer


class TrainAutoencoder(Trainer):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_display_batch: Optional[torch.Tensor] = None

    def train_step(self, input_batch) -> torch.Tensor:
        if isinstance(input_batch, (list, tuple)):
            input_batch = input_batch[0]

        output_batch = self.model(self.transform_input_batch(input_batch))

        if isinstance(output_batch, (list, tuple)):
            output_batch = output_batch[0]

        if input_batch.shape != output_batch.shape:
            raise ValueError(
                f"shapes differ after autoencoding: in={input_batch.shape}, out={output_batch.shape}"
            )

        #def _transform(x):
        #    f = torch.fft.fft2(x)
        #    #return f.real + f.imag
        #    return torch.cat([f.real[:, 3:-3], f.imag[:, 3:-3]])

        #input_batch = _transform(input_batch)
        #output_batch = _transform(output_batch)

        #loss = F.kl_div(input_batch, output_batch.clamp_min(0))
        loss = self.loss_function(input_batch, output_batch)
        return loss

    def write_step(self):

        def _get_reconstruction(batch_iterable, transform: bool = False, max_count: int = 32):
            images = []
            count = 0
            for batch in batch_iterable:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                images.append(batch)
                count += batch.shape[0]
                if count >= max_count:
                    break
            images = torch.cat(images)[:max_count].to(self.device)

            if transform:
                images = self.transform_input_batch(images)

            output_batch = self.model.forward(images)
            if isinstance(output_batch, (list, tuple)):
                output_batch = output_batch[0]

            output_batch = output_batch.clamp(0, 1)

            grid_images = []
            for i in range(0, images.shape[0], 8):
                for j in range(8):
                    if i + j < images.shape[0]:
                        grid_images.append(images[i + j])
                for j in range(8):
                    if i + j < images.shape[0]:
                        grid_images.append(output_batch[i + j])

            return images, output_batch, make_grid(grid_images, nrow=8)

        images, output_batch, grid = _get_reconstruction(self.iter_training_batches(), transform=True)
        self.log_image("train_reconstruction", grid)

        images, output_batch, grid = _get_reconstruction(self.iter_validation_batches())
        self.log_image("validation_reconstruction", grid)

        if hasattr(self.model, "encode"):
            features = self.model.encode(images)
        else:
            features = self.model.encoder(images)

        if isinstance(features, (list, tuple)):
            features = features[0]

        if "int" in str(features.dtype):
            features = features.to(images.dtype)

        self.log_image("validation_features", signed_to_image(features.flatten(1)))

        self.log_scalar("validation_features_mean", features.mean())
        self.log_scalar("validation_features_std", features.std())

        latent_shape = features.shape[1:]
        images = self.generate_random(latent_shape, 64, mean=features.mean(), std=features.std())
        self.log_image("image_random_generated", make_grid(images, nrow=8))

    @torch.no_grad()
    def generate_random(
            self,
            latent_shape: Tuple[int, ...],
            batch_size: int = 64,
            mean: float = 0.,
            std: float = 1.,
            seed: int = 32,
    ):
        gen = torch.Generator().manual_seed(seed)
        latent_batch = torch.randn(batch_size, *latent_shape, generator=gen).to(self.device) * std + mean

        if hasattr(self.model, "decode"):
            output_batch = self.model.decode(latent_batch)
        else:
            output_batch = self.model.decoder(latent_batch)

        if isinstance(output_batch, (list, tuple)):
            output_batch = output_batch[0]

        output_batch = output_batch.clamp(0, 1)

        return output_batch