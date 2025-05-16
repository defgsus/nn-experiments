import math
import random
import time
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import torch.utils.data
import torchvision.models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util.image import signed_to_image, get_images_from_iterable
from src.train import Trainer


class JigsawTrainer(Trainer):
    """
    Expects:
        dataset: (image-patches, target-class int)
        model:
            - forward(image-patches) == target-logits
    """
    def __init__(self, *args, num_classes: int, jigsaw_dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._jigsaw_dataset = jigsaw_dataset

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        input_patches, target_classes = input_batch

        target_logits = torch.zeros(input_patches.shape[0], self._num_classes, dtype=input_patches.dtype, device=input_patches.device)
        for b, cls in enumerate(target_classes):
            target_logits[b, cls] = 1.

        output_logits = self.model(input_patches)

        loss = F.cross_entropy(output_logits, target_logits)

        error_percent = (
            output_logits.argmax(dim=-1) != target_classes
        ).float().mean() * 100.

        return {
            "loss": loss,
            "error%": error_percent,
        }

    def write_step(self):
        #if not hasattr(self.data_loader.dataset, "_permutations") \
        #        or not hasattr(self.data_loader.dataset, "create_puzzle_crops"):
        #    return
        if self._jigsaw_dataset._permutations is None:
            self._jigsaw_dataset._permutations = self._jigsaw_dataset._create_permutations()

        grid = []
        crops, classes = next(self.iter_training_batches())
        for crop, cls in zip(crops, classes):
            grid.append(make_grid(crop, nrow=self._jigsaw_dataset._puzzle_size[-1]))

            predicted_class_logits = self.model(crop.to(self.device).unsqueeze(0))
            predicted_class = predicted_class_logits[0].argmax()

            def _plot_unpermuted(cls: int):
                perm = self._jigsaw_dataset._permutations[cls]
                ordered_crops = [None] * len(perm)
                for i, p in enumerate(perm):
                    ordered_crops[p] = crop[i]
                grid.append(make_grid(ordered_crops, nrow=self._jigsaw_dataset._puzzle_size[-1]))

            _plot_unpermuted(predicted_class)
            _plot_unpermuted(cls)

        self.log_image("image_puzzle_test", make_grid(grid, 9))
