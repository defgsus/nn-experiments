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


class TrainImg2Img(TrainAutoencoder):
    """
    expects (arg1, arg2) in data sample, then

<<<<<<< Updated upstream
        output = model(arg2)
        loss = arg1 - output

    if `first_arg_is_transforms` is defined, arg2 will be transformed from original arg1
=======
        output = model(arg1)
        loss = arg2 - output

    if `first_arg_is_transforms` is defined, arg1 will be transformed from original arg1
>>>>>>> Stashed changes
    """
    def __init__(
            self,
            *args,
            first_arg_is_transforms: List[Callable] = tuple(),
            histogram_loss_weight: float = 0.,
            image_loss_margin: int = 0,
            resize_log_images: float = 1.,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._first_arg_is_transforms = first_arg_is_transforms
        self.histogram_loss_weight = histogram_loss_weight
        self.histogram_loss = HistogramLoss(128)
        self.image_loss_margin = image_loss_margin
        self.resize_log_images = resize_log_images

    def _split_batch(self, batch):
        transformed_batch = None

        if isinstance(batch, (tuple, list)):
            # print("AAA", [b.shape if isinstance(b, torch.Tensor) else "X" for b in batch])
            if not self._first_arg_is_transforms:
                image_batch, transformed_batch = batch[:2]
            else:
                image_batch = batch[0]

        else:
            image_batch = batch

        if transformed_batch is None:
            assert self._first_arg_is_transforms, "Need to pass two images from dataset or define `first_arg_is_transforms`"
            transformed_batch = image_batch
            for transform in self._first_arg_is_transforms:
                transformed_batch = transform(transformed_batch)

        return image_batch, transformed_batch

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        image_batch, transformed_batch = self._split_batch(input_batch)

        if not hasattr(self, "_DUMPED_BATCH"):
            self._DUMPED_BATCH = True
            dump_module_stacktrace(self.model, transformed_batch)

        output_batch = self.model(transformed_batch)

        if image_batch.shape != output_batch.shape:
            raise ValueError(
                f"image_batch = {image_batch.shape}"
                f", output_batch = {output_batch.shape}"
            )

        if self.image_loss_margin > 0:
            image_batch = VF.crop(
                image_batch,
                self.image_loss_margin, self.image_loss_margin,
                image_batch.shape[-2] - 2 * self.image_loss_margin,
                image_batch.shape[-1] - 2 * self.image_loss_margin,
            )
            output_batch = VF.crop(
                output_batch,
                self.image_loss_margin, self.image_loss_margin,
                output_batch.shape[-2] - 2 * self.image_loss_margin,
                output_batch.shape[-1] - 2 * self.image_loss_margin,
            )

        loss_reconstruction_l1 = F.l1_loss(output_batch, image_batch)
        loss_reconstruction_l2 = F.mse_loss(output_batch, image_batch)

        losses = {
            "loss_reconstruction_l1": loss_reconstruction_l1,
            "loss_reconstruction_l2": loss_reconstruction_l2,
        }
        loss = loss_reconstruction_l1

        if self.histogram_loss_weight > 0:
            loss_histogram = self.histogram_loss(output_batch, image_batch)
            loss = loss + self.histogram_loss_weight * loss_histogram
            losses["loss_histogram"] = loss_histogram

        return {
            "loss": loss,
            **losses,
        }

    def write_step(self):

        def _get_reconstruction(batch_iterable, max_count: int = 32):
            images = []
            transformed_images = []
            count = 0
            for batch in batch_iterable:
                image_batch, transformed_batch = self._split_batch(batch)

                images.append(image_batch)
                transformed_images.append(transformed_batch)
                count += image_batch.shape[0]
                if count >= max_count:
                    break

            images = torch.cat(images)[:max_count].to(self.device)
            transformed_images = torch.cat(transformed_images)[:max_count].to(self.device)
            # print("XXX", images.shape, transformed_images.shape)
            restored_images = self.model(transformed_images).clamp(0, 1)

            max_channels = max(images.shape[-3], transformed_images.shape[-3])
            images = set_image_channels(images, max_channels)
            transformed_images = set_image_channels(transformed_images, max_channels)

            grid_images = []
            for i in range(images.shape[0]):
                grid_images.append(transformed_images[i])
                grid_images.append(images[i])
                grid_images.append(restored_images[i])

            grid = make_grid(grid_images, nrow=3 * 4)
            if self.resize_log_images != 1.:
                grid = VF.resize(
                    grid,
                    [int(s * self.resize_log_images) for s in grid.shape[-2:]],
                    VF.InterpolationMode.NEAREST, antialias=False,
                )
            return images, restored_images, grid

        images, restored_images, grid = _get_reconstruction(self.iter_training_batches())
        self.log_image("image_train_reconstruction", grid)

        images, restored_images, grid = _get_reconstruction(self.iter_validation_batches())
        self.log_image("image_validation_reconstruction", grid)
