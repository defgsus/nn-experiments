import math
import random
import itertools
import shutil
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import torch
import torch.nn
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_display_batch: Optional[torch.Tensor] = None

    def train_step(self, input_batch) -> torch.Tensor:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]
        input_batch = input_batch.to(self.device)

        output_batch = self.model.forward(input_batch)
        loss = F.mse_loss(input_batch, output_batch)

        return loss

    def write_step(self):
        if self.train_display_batch is None:
            images = get_images_from_iterable(self.data_loader)
            self.train_display_batch = torch.cat(images).to(self.device)

        output_batch1 = self.model.forward(self.validation_batch)
        output_batch2 = self.model.forward(self.train_display_batch)

        output_batch1 = output_batch1.clamp(0, 1)
        output_batch2 = output_batch2.clamp(0, 1)
        image = make_grid(
            [i for i in self.validation_batch[:8]] + [i for i in output_batch1[:8]]
            + [i for i in self.train_display_batch] + [i for i in output_batch2],
            nrow=8,
        )
        self.log_image("validation_reconstruction", image)

