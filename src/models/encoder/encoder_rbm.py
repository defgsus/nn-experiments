import math
from collections import OrderedDict
from typing import List, Iterable, Tuple, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.models.cnn import Conv2dBlock
from src.util.image import set_image_channels, image_resize_crop
from src.util import to_torch_device
from src.models.rbm import RBM
from .base import Encoder2d


class BoltzmanEncoder2d(Encoder2d):
    """
    A stack of Restricted Boltzman Machines
    """
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
            hidden_size: Iterable[int] = (),
            dropout: float = 0.,
            train_max_similarity: float = 0.,
    ):
        super().__init__(shape, code_size)

        self._hidden = list(hidden_size)
        self._dropout = dropout
        code_sizes = [code_size] + self._hidden

        self.rbms = nn.Sequential()
        for code_size, next_code_size in zip([math.prod(shape)] + code_sizes, code_sizes):
            self.rbms.append(
                RBM(code_size, next_code_size, dropout=dropout, train_max_similarity=train_max_similarity)
            )

    @property
    def device(self):
        return self.rbms[0].weight.device

    def forward(self, x):
        for rbm in self.rbms:
            x = rbm(x)
        return x

    def train_step(self, input_batch) -> torch.Tensor:
        """Combine loss of all RBMs"""
        data = input_batch
        if isinstance(data, (tuple, list)):
            data = data[0]

        loss = None
        for i, rbm in enumerate(self.rbms):
            loss_ = rbm.train_step(data)
            if loss is None:
                loss = loss_
            else:
                if isinstance(loss, dict):
                    for key, value in loss.items():
                        loss[key] = loss[key] + loss_[key]
                else:
                    loss = loss + loss_

            if i < len(self.rbms) - 1:
                data = rbm.forward(data)

        return loss

    def weight_images(self, **kwargs):
        return self.rbms[0].weight_images(**kwargs)

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "hidden_size": self._hidden,
            "dropout": self._dropout,
            "train_max_similarity": self.rbms[0].train_max_similarity,
        }

    @classmethod
    def from_data(cls, data: dict):
        extra = data["_extra_state"]
        model = cls(
            shape=extra["shape"],
            code_size=extra["code_size"],
            hidden_size=extra.get("hidden_size") or extra.get("hidden") or [],
            dropout=extra.get("dropout") or 0.,
            train_max_similarity=extra.get("train_max_similarity") or 0.,
        )
        # backward-compat
        if "rbm.weight" in data:
            data = {
                "_extra_state": data["_extra_state"],
                "rbms.0.weight": data["rbm.weight"],
                "rbms.0.bias_visible": data["rbm.bias_visible"],
                "rbms.0.bias_hidden": data["rbm.bias_hidden"],
            }
        model.load_state_dict(data)
        return model
