import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torchvision.models
from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
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
from src.train import Trainer
from src.models.cnn import *
from src.models.generative import *
from src.datasets import *
from src.util.image import *
from src.util.embedding import normalize_embedding
from src.util import num_module_parameters
from src.algo import Space2d
from src.models.vae import *
from src.models.transform import *
from src.algo import AudioUnderstander

from scripts.train_classifier_dataset import AlexNet
from scripts import datasets


class SimpleEncoder(nn.Module):
    def __init__(
            self,
            dimensions: Iterable[int],
            act_fn: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.dimensions = tuple(dimensions)

        self.layers = nn.Sequential()
        for i, (dim, next_dim) in enumerate(zip(self.dimensions, self.dimensions[1:])):
            self.layers.append(nn.Linear(dim, next_dim))
            if i < len(self.dimensions) - 2:
                self.layers.append(act_fn)

    def forward(self, batch):
        # print("XXX", batch)
        return self.layers(batch.flatten(1))

    def weight_images(self, **kwargs):
        images = []
        for layer in self.layers:
            if hasattr(layer, "weight"):
                images.append(layer.weight[:32])

        return images


class ContrastiveTrainer(Trainer):

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        features1 = self.model(input_batch[0])
        features2 = self.model(input_batch[1])
        target_dots = input_batch[2].to(features1.dtype) * 2. - 1.

        # constrain mean around zero
        mean_loss = features1.mean().abs()

        # constrain deviation below one
        std = features1.std()
        std_loss = (std - 1.).clamp_min(0)

        features1 = normalize_embedding(features1)
        features2 = normalize_embedding(features2)

        dots = torch.sum(features1 * features2, dim=-1)  # [N]
        dot_loss = F.mse_loss(dots, target_dots)

        return {
            "loss": (
                .1 * mean_loss
                + .2 * std_loss
                + dot_loss
            ),
            "loss_similarity": dot_loss,
            "loss_std": std_loss,
            "loss_mean": mean_loss,
        }


def mask_similar_transform(
        vector1: torch.Tensor, vector2: torch.Tensor, is_same: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    vectors = [vector1, vector2]
    # assert all(v.ndim == 1 for v in vectors), f"Got {[v.shape for v in vectors]}"

    if is_same:
        idx = random.randrange(2)
        v = vectors[idx][:]
        len = random.randrange(v.shape[-1] // 3)
        pos = random.randrange(v.shape[-1] - len)
        v[..., pos:pos + len] = 0
        vectors[idx] = v

    return tuple(vectors)


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (1, 256 * 3)
    ds = TensorDataset(
        torch.load("./datasets/embeddings-au-1sec-3x256.pt"),
        torch.load("./datasets/embeddings-au-1sec-3x256-ids.pt"),
    )
    ds = TransformDataset(ds, transforms=[lambda x: x.view(SHAPE)])

    sample = next(iter(ds))
    assert sample[0].shape == SHAPE, sample[0].shape

    num_test = 2000
    num_train = len(ds) - num_test
    train_ds, test_ds = torch.utils.data.random_split(ds, [num_train, num_test], torch.Generator().manual_seed(42))
    print(f"{len(test_ds)} validation samples")

    train_ds = IterableShuffle(train_ds, 50_000)

    train_ds = ContrastiveIterableDataset(
        train_ds, contrastive_ratio=.3, transform_ratio=.1, transforms=[mask_similar_transform]
    )
    test_ds = ContrastiveIterableDataset(test_ds)

    model = SimpleEncoder((math.prod(SHAPE), 64))
    print(model)

    #for i, (f1, f2, sim) in zip(range(20), DataLoader(train_ds, batch_size=128)):
    #    print(f"similars: {sim.sum()} / {sim.shape[0]} / {1. - sim.sum() / sim.shape[0]:.2f}")
    #return

    trainer = ContrastiveTrainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        num_inputs_between_validations=1_000_000 if isinstance(train_ds, IterableDataset) else None,
        data_loader=DataLoader(train_ds, batch_size=1024, num_workers=0, shuffle=not isinstance(train_ds, IterableDataset)),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        training_noise=.01,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=0.000001),
            #torch.optim.Adadelta(model.parameters(), lr=.1),
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
