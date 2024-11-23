import math
import dataclasses
from typing import Tuple

import torch
import torch.nn as nn

from src.models.util import *
from experiments.denoise.restorehalf_resconv import ResConvLayers
from .trainer import DiffusionModelInput, DiffusionModelOutput


class DiffusionModel(nn.Module):

    def __init__(
            self,
            image_channels: int,
            hidden_channels: int,
            num_layers: int,
            attention: int = 0,
            attention_heads: int = 0,
            norm=None,
    ):
        super().__init__()

        self.layers = ResConvLayers(
            channels_in=image_channels + 1,
            channels_out=image_channels,
            channels_hidden=hidden_channels,
            num_layers=num_layers,
            attention=attention,
            attention_heads=attention_heads,
            norm=norm,
        )
        #self.layers = nn.Sequential()
        #self.layers.add_module("input", nn.Conv2d(image_channels + 1, hidden_channels, 3, padding=1))
        #for i in range(num_layers):
        #    self.layers.add_module(f"conv_{i}", ResidualAdd(
        #        nn.Sequential(
        #            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        #            nn.GELU(),
        #        )
        #    ))
        #self.layers.add_module("output", nn.Conv2d(hidden_channels, image_channels, 3, padding=1))
        # self.layers.add_module("sig", nn.Sigmoid())

    def forward(self, input: DiffusionModelInput) -> DiffusionModelOutput:

        images_and_params = torch.concat([
            input.images,
            input.parameter_embedding()
        ], dim=-3)

        noise_prediction = self.layers(images_and_params)

        return DiffusionModelOutput(
            noise=noise_prediction,
        )
