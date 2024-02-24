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
from src.models.transform import *



class TrainDenoising(TrainAutoencoder):

    def __init__(
            self,
            *args,
            train_input_transforms=None,
            second_data_is_noise: bool = False,
            **kwargs,
    ):
        self._second_data_is_noise = second_data_is_noise
        if train_input_transforms is None and not second_data_is_noise:
            train_input_transforms = [
                ImageNoise(),
            ]
        super().__init__(*args, **kwargs, train_input_transforms=train_input_transforms)

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        transformed_batch = None
        if isinstance(input_batch, (tuple, list)):
            if self._second_data_is_noise:
                transformed_batch = input_batch[1]
            input_batch = input_batch[0]

        if transformed_batch is None:
            if self._second_data_is_noise:
                raise ValueError("Didn't get noisy 2nd batch from training set")

            transformed_batch = self.transform_input_batch(input_batch)

        if input_batch.shape != transformed_batch.shape:
            raise ValueError(
                f"input_batch = {input_batch.shape}"
                f", transformed_batch = {transformed_batch.shape}"
            )

        output_batch = self.model(transformed_batch)

        if input_batch.shape != output_batch.shape:
            raise ValueError(
                f"input_batch = {input_batch.shape}"
                f", output_batch = {output_batch.shape}"
            )

        reconstruction_loss = self.loss_function(output_batch, input_batch)
        # print("XX", reconstruction_loss, input_batch.mean(), transformed_batch.mean(), output_batch.mean())
        return {
            "loss": reconstruction_loss,
            "loss_reconstruction": reconstruction_loss,
        }

    def write_step(self):

        def _get_reconstruction(batch_iterable, transform: bool = False, max_count: int = 32):
            images = []
            transformed_images = []
            count = 0
            for batch in batch_iterable:
                if isinstance(batch, (list, tuple)):
                    if self._second_data_is_noise and transform:
                        transformed_images.append(batch[1])
                    batch = batch[0]

                images.append(batch)
                count += batch.shape[0]
                if count >= max_count:
                    break
            images = torch.cat(images)[:max_count].to(self.device)
            if transformed_images:
                transformed_images = torch.cat(transformed_images)[:max_count].to(self.device)
            else:
                transformed_images = None

            original_images = images

            if transform:
                if transformed_images is None:
                    images = self.transform_input_batch(images)
                else:
                    images = transformed_images

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
                for j in range(8):
                    if i + j < images.shape[0]:
                        grid_images.append(original_images[i + j])

            return images, output_batch, make_grid(grid_images, nrow=8)

        images, output_batch, grid = _get_reconstruction(self.iter_training_batches(), transform=True)
        self.log_image("train_reconstruction", grid)

        images, output_batch, grid = _get_reconstruction(self.iter_validation_batches(), transform=True)
        self.log_image("validation_reconstruction", grid)
