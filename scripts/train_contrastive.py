import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable


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
from src.train.trainer import Trainer
from src.models.cnn import *
from src.models.generative import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters
from src.algo import Space2d

from scripts.train_classifier_dataset import AlexNet


class ContrastiveImageTrainer(Trainer):

    def train_step(self, input_batch) -> torch.Tensor:
        features1 = self.model(input_batch[0])
        features2 = self.model(input_batch[1])
        target_dots = input_batch[2].to(features1.dtype) * 2. - 1.

        # constrain mean around zero
        mean_loss = features1.mean().abs()

        # constrain deviation below one
        std = features1.std()
        std_loss = (std - 1.).clamp_min(0)

        features1 = features1 / torch.norm(features1, dim=-1, keepdim=True)
        features2 = features2 / torch.norm(features2, dim=-1, keepdim=True)

        dots = torch.sum(features1 * features2, dim=-1)  # [N]
        # 1  1   0
        # 1  .5  .5
        # 1  -1  2
        # 0  1   2
        # 0  .5  1.5
        # 0  -1  0
        dot_loss = F.mse_loss(dots, target_dots)

        return (
            .1 * mean_loss
            + .2 * std_loss
            + dot_loss
        )

    @torch.no_grad()
    def write_step(self):
        feature_batch = []
        count = 0
        for input_batch in self.iter_validation_batches():
            features = self.model(input_batch[0].to(self.device))
            feature_batch.append(features)
            count += features.shape[0]
            if count >= 100:
                break

        features = torch.cat(feature_batch)
        self.log_image("validation_features", features.unsqueeze(0))
        self.log_scalar("validation_features_mean", features.mean())
        self.log_scalar("validation_features_std", features.std())

        features = features / torch.norm(features, dim=-1, keepdim=True)

        def _get_similar_indices(feat, count: int = 10):
            #feat = feat / feat.norm(dim=-1, keepdim=True)
            dot = feat @ features.T
            _, indices = torch.sort(dot, descending=True)
            #print("YYY", indices.shape, indices)
            return indices[:, :count]

        indices = list(range(10))
        sim_indices = _get_similar_indices(torch.cat([
            features[i].unsqueeze(0) for i in indices
        ]))

        images = [self.validation_sample(i)[0] for i in sim_indices.T.reshape(-1)]
        self.log_image("validation_similars", make_grid(images, nrow=len(indices)))


class EncoderTrans(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int = 128,
            patch_size: int = 32,
            width: int = 256,
            layers: int = 10,
            heads: int = 256 // 64,
    ):
        assert shape[-2] == shape[-1], shape

        super().__init__()
        self.encoder = VisionTransformer(
            shape[-1], patch_size=patch_size, width=width, layers=layers, heads=heads, output_dim=code_size
        )

    def forward(self, x):
        return self.encoder(x)

    def weight_images(self, **kwargs):
        images = []

        for resblock in self.encoder.transformer.resblocks:
            images.append(resblock.attn.out_proj.weight)
            images.append(resblock.mlp.c_fc.weight)
            images.append(resblock.mlp.c_proj.weight.T)

        if 0:
            weight = self.encoder.conv1.weight
            for w in weight[0, :10]:
                images.append(w)
            for w in weight[:10]:
                images.append(w[0])
        return images


class EncoderMLP(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int] = (128, ),
    ):
        assert shape[-2] == shape[-1], shape
        self.shape = shape

        super().__init__()
        channels = [math.prod(shape), *channels]
        self.layers = nn.Sequential(nn.Flatten(1))
        for size, next_size in zip(channels, channels[1:]):
            self.layers.append(
                nn.Linear(size, next_size)
            )

    def forward(self, x):
        return self.layers(x)

    def weight_images(self, **kwargs):
        images = []

        weight = self.layers[1].weight.reshape(-1, *self.shape)
        for w in weight[:10]:
            images.append(w)

        return images


class EncoderConv(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int = 128,
            kernel_size: int = 7,
            pool_kernel_size: int = 0,
            pool_type: str = "max",  # "average"
            stride: int = 1,
            channels: Iterable[int] = (8, 16, 32),
            batch_norm: bool = False,
    ):
        assert shape[-2] == shape[-1], shape

        super().__init__()
        channels = list(channels)
        self.conv = Conv2dBlock(
            channels=[shape[0], *channels],
            kernel_size=kernel_size,
            stride=stride,
            pool_kernel_size=pool_kernel_size,
            pool_type=pool_type,
            batch_norm=batch_norm,
        )
        out_shape = self.conv.get_output_shape(shape)
        self.linear = nn.Linear(math.prod(out_shape), code_size)
        #self.act = nn.ReLU()

    def forward(self, x):
        return self.linear(self.conv(x).view(x.shape[0], -1))
        #return self.act(y)

    def weight_images(self, **kwargs):
        images = []

        for layer in self.conv.layers:
            if hasattr(layer, "weight"):
                weight = layer.weight
                if weight.ndim == 4:
                    for w in weight[0, :32]:
                        images.append(w)
                    for w in weight[:32]:
                        images.append(w[0])
        return images


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    if 0:
        ORG_SHAPE = (3, 128, 128)
        SHAPE = (3, 64, 64)
        ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{ORG_SHAPE[-2]}x{ORG_SHAPE[-1]}.pt")[:1000])
        ds = TransformDataset(
            ds, dtype=torch.float, multiply=1. / 255.,
            #transforms=[VT.Grayscale()],
        )
    elif 0:
        ORG_SHAPE = (1, 128, 128)
        SHAPE = (3, 128, 128)
        ds = TensorDataset(
            torch.load(f"./datasets/pattern-{ORG_SHAPE[-3]}x{ORG_SHAPE[-2]}x{ORG_SHAPE[-1]}-uint.pt")
            [:1000]
        )
        ds = TransformDataset(
            ds, dtype=torch.float, multiply=1. / 255.,
            transforms=[
                #VT.Grayscale(),
                lambda i: i.repeat(3, 1, 1)
            ],
        )
        assert ds[0][0].shape == torch.Size(SHAPE), ds[0][0].shape

    else:
        ORG_SHAPE = (1, 128, 128)
        SHAPE = (3, 112, 112)
        ds = TransformDataset(
            TensorDataset(
                torch.load(f"./datasets/ifs-{ORG_SHAPE[-3]}x{ORG_SHAPE[-2]}x{ORG_SHAPE[-1]}-uint8-1000x32.pt"),
                torch.load(f"./datasets/ifs-{ORG_SHAPE[-3]}x{ORG_SHAPE[-2]}x{ORG_SHAPE[-1]}-uint8-1000x32-labels.pt")
            ),
            dtype=torch.float, multiply=1. / 255.,
            transforms=[
                #VT.RandomRotation(20),
                #VT.RandomCrop(SHAPE[-2:]),
                lambda i: i.repeat(3, 1, 1),
            ],
            # num_repeat=5,
        )
        assert ds[0][0].shape == torch.Size((SHAPE[0], *ORG_SHAPE[-2:])), ds[0][0].shape

    ds = ContrastiveImageDataset(
        ds, crop_shape=SHAPE[-2:],
        num_crops=3, num_contrastive_crops=3,
        prob_h_flip=.5,
        prob_v_flip=.5,
        prob_hue=.0,
        prob_saturation=0.,
        prob_brightness=0.5,
        prob_grayscale=1.,
        use_labels=True,
    )
    assert ds[0][0].shape == torch.Size(SHAPE), ds[0][0].shape

    num_valid = 2000
    num_train = len(ds) - num_valid
    train_ds, test_ds = torch.utils.data.random_split(ds, [num_train, num_valid], torch.Generator().manual_seed(42))
    print(f"{len(test_ds)} validation samples")

    CODE_SIZE = 128
    #model = EncoderTrans(SHAPE, code_size=128)
    #model = EncoderMLP(SHAPE, channels=[64, 2])
    model = EncoderConv(SHAPE, code_size=2, channels=[32, 32, 64], kernel_size=16, pool_kernel_size=16, batch_norm=True)#, pool_type="average")
    #model = EncoderConv(SHAPE, code_size=2, channels=[32, 32], kernel_size=16, max_pool_kernel_size=8)
    model = AlexNet(num_classes=CODE_SIZE)
    print(model)

    trainer = ContrastiveImageTrainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        #max_inputs=100_000_000_000,
        max_epoch=1000,
        #data_loader=DataLoader(train_ds, shuffle=True, batch_size=10),
        data_loader=DataLoader(train_ds, batch_size=128),# num_workers=5),
        validation_loader=DataLoader(test_ds, batch_size=50),
        freeze_validation_set=True,
        optimizers=[
            #torch.optim.SGD(model.parameters(), lr=.001, weight_decay=0.00001),
            #torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.001),
            torch.optim.Adadelta(model.parameters(), lr=.2),
            #torch.optim.RMSprop(model.parameters(), lr=.001)
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
