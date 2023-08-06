import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable


from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid
from clip.model import VisionTransformer

from src import console
from src.train.trainer import Trainer
from src.models.cnn import *
from src.datasets import *


class TransformerModel(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int = 128,
            feature_size: int = 8,
            patch_size: int = 32,
            width: int = 256,
            layers: int = 10,
            heads: int = 8,
    ):
        assert shape[-2] == shape[-1], shape

        super().__init__()

        self.encoder = VisionTransformer(
            shape[-1], patch_size=patch_size, width=width, layers=layers, heads=heads, output_dim=code_size
        )
        self.linear = nn.Linear(code_size, feature_size)

    def forward(self, x):
        return self.linear(self.encoder(x))

    def train_step(self, input_batch) -> torch.Tensor:
        image_batch, feature_batch = input_batch
        output_batch = self(image_batch)

        return F.l1_loss(output_batch, feature_batch)


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (3, 64, 64)
    dataset = TensorDataset(torch.load(f"./datasets/kali-uint8-{SHAPE[-2]}x{SHAPE[-1]}.pt"))
    dataset = TransformDataset(
        dataset, dtype=torch.float, multiply=1. / 255.,
        features_dataframe=pd.read_pickle(f"./datasets/kali-uint8-{SHAPE[-2]}x{SHAPE[-1]}-features.df")
    )
    assert dataset[0][0].shape[:3] == torch.Size(SHAPE), dataset[0][0].shape

    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.99, 0.01], torch.Generator().manual_seed(42))

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    model = TransformerModel(SHAPE, feature_size=dataset[0][1].shape[-1])
    print(model)

    trainer = Trainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_inputs_between_validations=100_000,
        #data_loader=DataLoader(train_ds, shuffle=True, batch_size=10),
        data_loader=DataLoader(train_ds, batch_size=50),
        validation_loader=DataLoader(test_ds),
        optimizers=[
            #torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0.001),
            torch.optim.Adadelta(model.parameters(), lr=1.),
        ],
        hparams={
            "shape": SHAPE,
        },
        weight_image_kwargs={
            "shape": SHAPE,
        }
    )

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


if __name__ == "__main__":
    main()
