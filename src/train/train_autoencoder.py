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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_display_batch: Optional[torch.Tensor] = None

    def train_step(self, input_batch) -> torch.Tensor:
        input_batch = input_batch[0]

        output_batch = self.model.forward(input_batch)

        def _transform(x):
            f = torch.fft.fft2(x)
            #return f.real + f.imag
            return torch.cat([f.real[:, 3:-3], f.imag[:, 3:-3]])

        #input_batch = _transform(input_batch)
        #output_batch = _transform(output_batch)

        #loss = F.kl_div(input_batch, output_batch.clamp_min(0))
        loss = F.l1_loss(input_batch, output_batch)
        return loss

    def write_step(self):
        images = []
        count = 0
        for batch in self.iter_validation_batches():
            images.append(batch[0])
            count += batch[0].shape[0]
            if count >= 32:
                break
        images = torch.cat(images)[:32].to(self.device)

        output_batch = self.model.forward(images).clamp(0, 1)
        grid_images = []
        for i in range(0, images.shape[0], 8):
            for j in range(8):
                if i + j < images.shape[0]:
                    grid_images.append(images[i + j])
            for j in range(8):
                if i + j < images.shape[0]:
                    grid_images.append(output_batch[i + j])

        image = make_grid(grid_images, nrow=8)
        self.log_image("validation_reconstruction", image)

