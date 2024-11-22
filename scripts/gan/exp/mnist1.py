import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.models.util import *
from src.models.transform import *
from src.datasets import *
from experiments.datasets import *
from scripts.gan.trainer import GANTrainerSettings, DiscriminatorOutput


SHAPE = (1, 28, 28)
CODE_SIZE = 32
NUM_CLASSES = 10


def get_settings() -> GANTrainerSettings:
    return GANTrainerSettings(
        code_size=CODE_SIZE,
        num_classes=NUM_CLASSES,
        loss_type="class",
        steps_per_module=64 * 10,
        #steps_per_discriminator_ratio=2,
        discriminator_bootstrap_steps=64 * 1000,
        #switch_if_loss_below=True,
    )


def get_dataset(validation: bool):
    return ClassLogitsDataset(
        mnist_dataset(train=not validation, shape=SHAPE),
        num_classes=NUM_CLASSES, tuple_position=1, label_to_index=True,
    )


def get_generator() -> nn.Module:

    class Generator(nn.Module):

        def __init__(self):
            super().__init__()

            hidden_shape = (32, *SHAPE[-2:])

            self.layers = nn.Sequential()
            self.layers.add_module("input", nn.Linear(CODE_SIZE, math.prod(hidden_shape)))
            self.layers.add_module("reshape", Reshape(hidden_shape))
            for i in range(5):
                self.layers.add_module(f"conv_{i}", ResidualAdd(
                    nn.Sequential(
                        nn.Conv2d(hidden_shape[0], hidden_shape[0], 3, padding=1),
                        nn.GELU(),
                    )
                ))

            self.layers.add_module("output", nn.Conv2d(hidden_shape[0], SHAPE[0], 3, padding=1))
            self.layers.add_module("sig", nn.Sigmoid())

        def forward(self, x):
            return self.layers(x)

    return Generator()


def get_discriminator() -> nn.Module:

    class Discriminator(nn.Module):

        def __init__(self):
            super().__init__()

            hidden_shape = (32, *SHAPE[-2:])

            self.layers = nn.Sequential()
            self.layers.add_module("input", nn.Conv2d(SHAPE[0], hidden_shape[0], 3, padding=1))

            for i in range(5):
                self.layers.add_module(f"conv_{i}", ResidualAdd(
                    nn.Sequential(
                        nn.Conv2d(hidden_shape[0], hidden_shape[0], 3, padding=1),
                        nn.GELU(),
                    )
                ))
            self.layers.add_module("flatten", nn.Flatten(-3))
            self.layers.add_module("proj", nn.Linear(math.prod(hidden_shape), CODE_SIZE))

            # self.output_critic = nn.Linear(CODE_SIZE, 1)
            self.output_label = nn.Linear(CODE_SIZE, NUM_CLASSES + 1)
            #self.layers.add_module("act", nn.Tanh())

        def forward(self, x):
            y = self.layers(x)
            return DiscriminatorOutput(
                # critic=self.output_critic(y),
                class_logits=self.output_label(y),
            )

    class Discriminator2(nn.Module):

        def __init__(self):
            super().__init__()

            hidden_shape = (32, *SHAPE[-2:])

            self.layers = nn.Sequential()
            self.layers.add_module("input", nn.Conv2d(SHAPE[0], hidden_shape[0], 3, padding=1))

            size = SHAPE[-1]
            channels = hidden_shape[0]
            idx = -1
            while size > 4 and size % 2 == 0:
                idx += 1
                self.layers.append(ResidualAdd(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.GELU(),
                    )
                ))
                if idx % 3 == 2:
                    self.layers.append(nn.PixelUnshuffle(2))
                    size //= 2
                    channels *= 4
                #else:
                #    self.layers.append(nn.Conv2d(channels, channels, 5))
                #    size -= 4

            self.layers.add_module("flatten", nn.Flatten(-3))
            self.layers.add_module("proj", nn.Linear(channels * size * size, CODE_SIZE))

            # self.output_critic = nn.Linear(CODE_SIZE, 1)
            self.output_label = nn.Linear(CODE_SIZE, NUM_CLASSES + 1)
            #self.layers.add_module("act", nn.Tanh())

        def forward(self, x):
            y = self.layers(x)
            return DiscriminatorOutput(
                # critic=self.output_critic(y),
                class_logits=self.output_label(y),
            )

    from torchvision.models import resnet
    class ResNetDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = resnet.resnet50(num_classes=NUM_CLASSES + 1)

        def forward(self, x):
            return DiscriminatorOutput(
                class_logits=self.resnet(x)
            )

    return Discriminator2()
    #return ResNetDiscriminator()

