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
from .model import BDModelInput, BDModelOutput
from .datasets import BDEntry


class TrainBoulderDashPredict(Trainer):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _model_forward(self, input_batch) -> BDModelOutput:
        return self.model(input_batch, **self.model_forward_kwargs)

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (list, tuple)):
            input_batch = input_batch[0]

        output_batch = self._model_forward(input_batch)

        #if False and not hasattr(self, "_printed_"):
        #    self._printed_ = True
        #    print("EXPECTED: ", map2_batch[0])
        #    print("PREDICTED:", map2_batch_out[0])

        return self._calc_loss(input_batch, output_batch)

    def _calc_loss(self, expected: BDEntry, predicted: BDModelOutput):
        num_obj = BoulderDash.OBJECTS.count()

        if 0:
            object_loss = self.loss_function(expected[:, :num_obj, :, :], predicted[:, :num_obj, :, :])
            state_loss = self.loss_function(expected[:, num_obj:, :, :], predicted[:, num_obj:, :, :])

        else:
            B, C, H, W = expected.state.shape
            expected_state = expected.next_state.permute(0, 2, 3, 1).reshape(B * H * W, -1)
            predicted_state = predicted.next_state.permute(0, 2, 3, 1).reshape(B * H * W, -1)

            func = F.soft_margin_loss

            object_loss = func(predicted_state[:, :num_obj], expected_state[:, :num_obj])
            state_loss = func(predicted_state[:, num_obj:], expected_state[:, num_obj:])

            if predicted.reward is not None:
                reward_loss = F.l1_loss(predicted.reward, expected.reward)
            else:
                reward_loss = torch.tensor(0).to(self.device)

        return {
            "loss": (object_loss + state_loss) / 2 + reward_loss,
            "loss_object": object_loss,
            "loss_state": state_loss,
            "loss_reward": reward_loss,
        }

    def validation_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (list, tuple)):
            input_batch = input_batch[0]

        output_batch = self._model_forward(input_batch)

        losses = self._calc_loss(input_batch, output_batch)

        next_state = torch.concat([
            BoulderDash.from_tensor(map).to_tensor().unsqueeze(0).to(output_batch.next_state.device)
            for map in input_batch.next_state
        ])
        predicted_next_state = torch.concat([
            BoulderDash.from_tensor(map).to_tensor().unsqueeze(0).to(output_batch.next_state.device)
            for map in output_batch.next_state
        ])

        correct_map = ((next_state - predicted_next_state).abs() < .1)
        accuracy = correct_map.float().mean() * 100

        return {
            **losses,
            "error": 100 - accuracy,
        }

    def write_step(self):

        def _get_prediction(
                batch_iterable, transform: bool = False, max_count: int = 8,
                size: int = 8,
        ):
            for batch in batch_iterable:
                break
            batch: BDEntry
            if batch.state is not None:
                batch.state = batch.state[:max_count]
            if batch.next_state is not None:
                batch.next_state = batch.next_state[:max_count]
            if batch.reward is not None:
                batch.reward = batch.reward[:max_count]
            if batch.action is not None:
                batch.action = batch.action[:max_count]

            if transform:
                batch.state = self.transform_input_batch(batch.state)

            output_batch = self._model_forward(batch.to(self.device))

            grid_maps = []
            grid_images = []
            for map, repro in zip(batch.next_state, output_batch.next_state):
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

