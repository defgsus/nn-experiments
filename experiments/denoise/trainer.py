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
from src.train.train_autoencoder import TrainAutoencoder
from src.models.transform import Sobel


def add_noise(
        x: torch.Tensor,
        amt_min: float = .01,
        amt_max: float = .15,
        amt_power: float = 2.,
):
    amt = math.pow(random.uniform(0, 1), amt_power)
    amt = amt_min + (amt_max - amt_min) * amt

    return x + amt * torch.randn_like(x)


class TrainDenoising(TrainAutoencoder):

    def __init__(self, *args, train_input_transforms=None, **kwargs):
        if train_input_transforms is None:
            train_input_transforms = [
                add_noise,
            ]
        super().__init__(*args, **kwargs, train_input_transforms=train_input_transforms)

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        transformed_batch = self.transform_input_batch(input_batch)

        if input_batch.shape != transformed_batch.shape:
            raise ValueError(
                f"input_batch = {transformed_batch.shape}"
                f", transformed_batch = {transformed_batch.shape}"
            )

        output_batch = self.model(transformed_batch)

        if input_batch.shape != output_batch.shape:
            raise ValueError(
                f"input_batch = {input_batch.shape}"
                f", output_batch = {output_batch.shape}"
            )

        reconstruction_loss = self.loss_function(output_batch, input_batch)

        return {
            "loss": reconstruction_loss,
            "loss_reconstruction": reconstruction_loss,
        }

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

        images, output_batch, grid = _get_reconstruction(self.iter_validation_batches(), transform=True)
        self.log_image("validation_reconstruction", grid)
