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
from src.util.image import set_image_channels
from src.util.module import dump_module_stacktrace
from src.models.loss import HistogramLoss


class TrainImg2ImgDiffusion(TrainAutoencoder):
    """
    expects (arg1, arg2, arg3) in data sample, then

        output = model(arg1, arg3)
        loss = arg2 - output

    """
    def __init__(
            self,
            *args,
            histogram_loss_weight: float = 0.,
            resize_log_images: float = 1.,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.histogram_loss_weight = histogram_loss_weight
        self.histogram_loss = HistogramLoss(128)
        self.resize_log_images = resize_log_images

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        source_batch, target_batch, amount_batch = input_batch

        if not hasattr(self, "_DUMPED_BATCH"):
            self._DUMPED_BATCH = True
            dump_module_stacktrace(self.model, source_batch, amount_batch)

        output_batch = self.model(source_batch, amount_batch)

        if target_batch.shape != output_batch.shape:
            raise ValueError(
                f"target_batch = {target_batch.shape}"
                f", output_batch = {output_batch.shape}"
            )

        loss_reconstruction_l1 = F.l1_loss(output_batch, target_batch)
        loss_reconstruction_l2 = F.mse_loss(output_batch, target_batch)

        losses = {
            "loss_reconstruction_l1": loss_reconstruction_l1,
            "loss_reconstruction_l2": loss_reconstruction_l2,
        }
        loss = loss_reconstruction_l1

        if self.histogram_loss_weight > 0:
            loss_histogram = self.histogram_loss(output_batch, target_batch)
            loss = loss + self.histogram_loss_weight * loss_histogram
            losses["loss_histogram"] = loss_histogram

        return {
            "loss": loss,
            **losses,
        }

    def write_step(self):

        def _get_reconstruction(batch_iterable, max_count: int = 32):
            source_images = []
            target_images = []
            amounts = []
            count = 0
            for batch in batch_iterable:
                source_batch, target_batch, amount_batch = batch

                source_images.append(source_batch)
                target_images.append(target_batch)
                amounts.append(amount_batch)
                count += source_batch.shape[0]
                if count >= max_count:
                    break

            source_images = torch.cat(source_images)[:max_count].to(self.device)
            target_images = torch.cat(target_images)[:max_count].to(self.device)
            amounts = torch.cat(amounts)[:max_count].to(self.device)
            restored_images = self.model(source_images, amounts).clamp(0, 1)

            max_channels = max(source_images.shape[-3], target_images.shape[-3])
            source_images = set_image_channels(source_images, max_channels)
            target_images = set_image_channels(target_images, max_channels)

            grid_images = []
            for i in range(source_images.shape[0]):
                grid_images.append(source_images[i])
                grid_images.append(target_images[i])
                grid_images.append(restored_images[i])

            grid = make_grid(grid_images, nrow=3 * 4)
            if self.resize_log_images != 1.:
                grid = VF.resize(
                    grid,
                    [int(s * self.resize_log_images) for s in grid.shape[-2:]],
                    VF.InterpolationMode.NEAREST, antialias=False,
                )
            return source_images, restored_images, grid

        images, restored_images, grid = _get_reconstruction(self.iter_training_batches())
        self.log_image("image_train_reconstruction", grid)

        images, restored_images, grid = _get_reconstruction(self.iter_validation_batches())
        self.log_image("image_validation_reconstruction", grid)
