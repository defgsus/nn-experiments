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

from src.util.image import signed_to_image, get_images_from_iterable
from src.train import Trainer
from src.algo.boulderdash import BoulderDash


class TrainBoulderDashPredict(Trainer):

    def __init__(self, *args, loss_crop: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_crop = loss_crop

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        map1_batch, map2_batch = input_batch
        map1_batch = self.transform_input_batch(map1_batch)

        map2_batch_out = self.model(map1_batch)

        if False and not hasattr(self, "_printed_"):
            self._printed_ = True
            print("EXPECTED: ", map2_batch[0])
            print("PREDICTED:", map2_batch_out[0])

        return self._calc_loss(map2_batch, map2_batch_out)

    def _crop(self, x):
        if self._loss_crop > 0:
            return x[..., self._loss_crop:-self._loss_crop, self._loss_crop:-self._loss_crop]
        return x

    def _calc_loss(self, expected, predicted):
        num_obj = BoulderDash.OBJECTS.count()
        expected = self._crop(expected)
        predicted = self._crop(predicted)

        if 0:
            object_loss = self.loss_function(expected[:, :num_obj, :, :], predicted[:, :num_obj, :, :])
            state_loss = self.loss_function(expected[:, num_obj:, :, :], predicted[:, num_obj:, :, :])

        else:
            B, C, H, W = expected.shape
            expected = expected.permute(0, 2, 3, 1).reshape(B * H * W, -1)
            predicted = predicted.permute(0, 2, 3, 1).reshape(B * H * W, -1)

            func = F.soft_margin_loss

            object_loss = func(predicted[:, :num_obj], expected[:, :num_obj])
            state_loss = func(predicted[:, num_obj:], expected[:, num_obj:])

        return {
            "loss": (object_loss + state_loss) / 2,
            "loss_object": object_loss,
            "loss_state": state_loss,
        }

    def validation_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        map1_batch, map2_batch = input_batch

        map2_batch_out = self.model(map1_batch)

        losses = self._calc_loss(map2_batch, map2_batch_out)

        map2_batch_thresh = self._crop(torch.concat([
            BoulderDash.from_tensor(map).to_tensor().unsqueeze(0).to(map2_batch_out.device)
            for map in map2_batch
        ]))
        map2_batch_out_thresh = self._crop(torch.concat([
            BoulderDash.from_tensor(map).to_tensor().unsqueeze(0).to(map2_batch_out.device)
            for map in map2_batch_out
        ]))

        accuracy = ((map2_batch_thresh - map2_batch_out_thresh).abs() < .1).sum() / math.prod(map2_batch.shape) * 100

        return {
            **losses,
            "error": 100 - accuracy,
        }

    def write_step(self):

        def _get_prediction(
                batch_iterable, transform: bool = False, max_count: int = 8,
                size: int = 8,
        ):
            count = 0
            map1_batch = []
            map2_batch = []
            for map1, map2 in batch_iterable:
                map1_batch.append(map1)
                map2_batch.append(map2)
                count += map1.shape[0]
                if count >= max_count:
                    break
            map1_batch = torch.cat(map1_batch)[:max_count].to(self.device)
            map2_batch = torch.cat(map2_batch)[:max_count].to(self.device)

            if transform:
                map1_batch = self.transform_input_batch(map1_batch)

            output_batch = self.model.forward(map1_batch)

            grid_maps = []
            grid_images = []
            for map, repro in zip(map2_batch, output_batch):
                grid_maps.append(map[:3])
                grid_maps.append(repro[:3])
                grid_maps.append(map[-3:])
                grid_maps.append(repro[-3:])
                grid_images.append(torch.from_numpy(BoulderDash.from_tensor(map).to_image(size)))
                grid_images.append(torch.from_numpy(BoulderDash.from_tensor(repro).to_image(size)))

            return (
                make_grid(grid_maps, nrow=4).clamp(0, 1),
                make_grid(grid_images, nrow=2, padding=size),
            )

        grid_map, grid_image = _get_prediction(self.iter_training_batches(), transform=True)
        self.log_image("train_prediction_map", grid_map)
        self.log_image("train_prediction_image", grid_image)

        grid_map, grid_image = _get_prediction(self.iter_validation_batches())
        self.log_image("validation_prediction_map", grid_map)
        self.log_image("validation_prediction_image", grid_image)
