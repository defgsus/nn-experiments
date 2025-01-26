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


class TrainAutoencoderSpecial(TrainAutoencoder):

    def __init__(
            self,
            *args,
            feature_loss_weight: float = 0.0001,
            perceptual_model: Optional[torch.nn.Module] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_loss_weight = feature_loss_weight
        self._perceptual_model = perceptual_model
        if perceptual_model is not None and callable(getattr(perceptual_model, "to", None)):
            self._perceptual_model.to(self.device)

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        transformed_batch = self.transform_input_batch(input_batch)

        if hasattr(self.model, "encode"):
            feature_batch = self.model.encode(transformed_batch)
        else:
            feature_batch = self.model.encoder(transformed_batch)

        if not self.extra_description_values:
            self.extra_description_values = {}
        if not self.extra_description_values.get("extra"):
            self.extra_description_values["extra"] = {}
        if self.extra_description_values["extra"].get("compression-ratio") is None:
            cr = math.prod(transformed_batch.shape) / math.prod(feature_batch.shape)
            self.extra_description_values["extra"]["compression-ratio"] = cr
            print("LATENT SHAPE:", feature_batch.shape[1:])
            print("COMPRESSION RATIO:", cr)

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

        perceptual_loss = None
        if self._perceptual_model:
            output_batch2 = self._perceptual_model(output_batch)
            input_batch2 = self._perceptual_model(input_batch)
            perceptual_loss = self.loss_function(output_batch2, input_batch2)

        reconstruction_loss = self.loss_function(output_batch, input_batch)

        # print("XX", float(reconstruction_loss), float(self.loss_function(output_batch, transformed_batch)), float(self.loss_function(input_batch, transformed_batch)))

        if 0:
            if not hasattr(self, "_sobel_filter"):
                self._sobel_filter = Sobel(direction=True).to(self.device)

            sobel_input_batch = self._sobel_filter(input_batch)
            sobel_output_batch = self._sobel_filter(output_batch)
            sobel_reconstruction_loss = self.loss_function(sobel_output_batch, sobel_input_batch)

        if "int" in str(feature_batch.dtype):
            feature_batch = feature_batch.to(input_batch.dtype)

        loss_batch_std = (.5 - feature_batch.std(0).mean()).abs()
        loss_batch_mean = (0. - feature_batch.mean()).abs()

        if perceptual_loss is not None:
            loss = perceptual_loss
        else:
            loss = reconstruction_loss

        if self.feature_loss_weight:
            loss = loss + self.feature_loss_weight * (loss_batch_std + loss_batch_mean)

        extra_loss = {}
        if callable(getattr(self.model, "extra_loss", None)):
            extra_loss = self.model.extra_loss() or {}
            for key, value in extra_loss.items():
                loss = loss + value

        return {
            "loss": loss,
            "loss_reconstruction": reconstruction_loss,
            "loss_perceptual": perceptual_loss,
            # "loss_reconstruction_sobel": sobel_reconstruction_loss,
            "loss_batch_std": loss_batch_std,
            "loss_batch_mean": loss_batch_mean,
            **extra_loss,
        }
