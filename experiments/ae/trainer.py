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


class TrainAutoencoderSpecial(TrainAutoencoder):

    def __init__(self, *args, feature_loss_weight: float = 0.0001, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_loss_weight = feature_loss_weight

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        if hasattr(self.model, "encode"):
            feature_batch = self.model.encode(input_batch)
        else:
            feature_batch = self.model.encoder(input_batch)

        if hasattr(self.model, "decode"):
            output_batch = self.model.decode(feature_batch)
        else:
            output_batch = self.model.decoder(feature_batch)

        if input_batch.shape != output_batch.shape:
            raise ValueError(
                f"input_batch = {input_batch.shape}"
                f", output_batch = {output_batch.shape}"
                f", feature_batch = {feature_batch.shape}"
            )

        reconstruction_loss = self.loss_function(output_batch, input_batch)

        if 0:
            if not hasattr(self, "_sobel_filter"):
                self._sobel_filter = Sobel()

            sobel_input_batch = self._sobel_filter(input_batch)
            sobel_output_batch = self._sobel_filter(output_batch)
            sobel_reconstruction_loss = self.loss_function(sobel_input_batch, sobel_output_batch)

        if "int" in str(feature_batch.dtype):
            feature_batch = feature_batch.to(input_batch.dtype)

        loss_batch_std = (.5 - feature_batch.std(0).mean()).abs()
        loss_batch_mean = (0. - feature_batch.mean()).abs()

        loss = reconstruction_loss
        if self.feature_loss_weight:
            loss = loss + self.feature_loss_weight * (loss_batch_std + loss_batch_mean)

        return {
            "loss": loss,
            "loss_reconstruction": reconstruction_loss,
            #"loss_reconstruction_sobel": sobel_reconstruction_loss,
            "loss_batch_std": loss_batch_std,
            "loss_batch_mean": loss_batch_mean,
        }
