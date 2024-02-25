import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Any

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader, TensorDataset

from src import console
from src.train.train_autoencoder import TrainAutoencoder
from src.models.cnn import Conv2dBlock
from src.datasets import ImageFolderIterableDataset, ImageAugmentation, IterableShuffle
from src.models.cnn import ConvAutoEncoder
from src.util.image import set_image_channels


class ResizeDataset(Dataset):

    def __init__(self, dataset: Dataset, shape: Tuple[int, int, int]):
        self.dataset = dataset
        self.shape = tuple(shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = self.dataset[item]
        if isinstance(image, tuple):
            return (self.apply(image[0]), ) + image[1:]
        elif isinstance(image, list):
            return [self.apply(image[0])] + image[1:]
        elif isinstance(image, torch.Tensor):
            return self.apply(image)
        else:
            return image

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return set_image_channels(VF.resize(image, size=list(self.shape[-2:])), self.shape[0])
        else:
            return image


class ResizeDatasetIterable(IterableDataset):

    def __init__(self, dataset: Dataset, shape: Tuple[int, int, int]):
        self.dataset = dataset
        self.shape = tuple(shape)

    def __iter__(self):
        for image in self.dataset:
            if isinstance(image, tuple):
                yield (self.apply(image[0]), ) + image[1:]
            elif isinstance(image, list):
                yield [self.apply(image[0])] + image[1:]
            elif isinstance(image, torch.Tensor):
                yield self.apply(image)
            else:
                yield image

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            i = set_image_channels(VF.resize(image, size=list(self.shape[-2:])), self.shape[0])
            # print("RESIZED", image.shape, i.shape, self.shape)
            return i
        else:
            return image


def first(iterable) -> Any:
    for i in iterable:
        return i


def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)

    kwargs = vars(parser.parse_args())
    reset = kwargs.pop("reset")

    model = None

    # ds = TensorDataset(torch.load("./datasets/diverse-64x64-aug4.pt"))
    train_ds = TensorDataset(torch.load("./datasets/diverse-32x32-aug16.pt"))
    validation_ds = ImageFolderIterableDataset(Path("~/Pictures/bob/").expanduser())

    steps_per_resolution = 500_000
    trainer = None
    for train_idx, (num_inputs, shape) in enumerate((
            (steps_per_resolution, (3, 8, 8)),
            (steps_per_resolution, (3, 12, 12)),
            (steps_per_resolution, (3, 16, 16)),
            (steps_per_resolution, (3, 20, 20)),
            (steps_per_resolution, (3, 24, 24)),
            (steps_per_resolution, (3, 28, 28)),
            (steps_per_resolution, (3, 32, 32)),
    )):
        reset = reset and train_idx == 0

        if model is None:
            model = ConvAutoEncoder(shape=shape, channels=[16])
        else:
            model.add_layer(channels=16)
            model.shape = shape
            # print(model)

        train_ds_resized = ResizeDataset(train_ds, shape)
        assert train_ds_resized[0][0].shape[-3:] == torch.Size(shape), train_ds_resized[0][0].shape
        validation_ds_resized = ResizeDatasetIterable(validation_ds, shape)
        assert first(validation_ds_resized).shape[-3:] == torch.Size(shape), first(validation_ds_resized).shape

        previous_trainer = trainer
        trainer = TrainAutoencoder(
            **kwargs,
            max_inputs=num_inputs,
            num_inputs_between_validations=20_000,
            reset=reset,
            model=model,
            data_loader=DataLoader(train_ds_resized, shuffle=True, batch_size=10),
            validation_loader=DataLoader(validation_ds_resized),
            optimizers=[
                #torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001),
                torch.optim.Adadelta(model.parameters(), lr=0.1),
            ]
        )
        if previous_trainer is not None:
            trainer.num_input_steps = previous_trainer.num_input_steps
            trainer.num_batch_steps = previous_trainer.num_batch_steps
            trainer.epoch = previous_trainer.epoch
            trainer.max_inputs += trainer.num_input_steps

        #if not reset:
        #    trainer.load_checkpoint()

        trainer.save_description()
        trainer.train()


if __name__ == "__main__":
    main()
