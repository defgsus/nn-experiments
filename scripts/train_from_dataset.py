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
import clip

from src import console
from src.train.trainer import Trainer
from src.models.cnn import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters


class EncoderMLP(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int] = (128, ),
            hidden_act: Optional[nn.Module] = None,
    ):
        assert shape[-2] == shape[-1], shape
        self.shape = shape

        super().__init__()
        channels = [math.prod(shape), *channels]
        self.layers = nn.Sequential(nn.Flatten(1))
        for i, (size, next_size) in enumerate(zip(channels, channels[1:])):
            self.layers.append(
                nn.Linear(size, next_size)
            )
            if hidden_act is not None and i < len(channels) - 2:
                self.layers.append(hidden_act)

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

        if 1:
            weight = self.encoder.conv1.weight
            for w in weight[0, :10]:
                images.append(w)
            for w in weight[:10]:
                images.append(w[0])

        for resblock in self.encoder.transformer.resblocks:
            images.append(resblock.attn.out_proj.weight)
            images.append(resblock.mlp.c_fc.weight)
            images.append(resblock.mlp.c_proj.weight.T)

        return images


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (3, 64, 64)
    CODE_SIZE = 512
    ds = ConcatDataset(
        [
            TransformDataset(
                TensorDataset(
                    torch.load(f"./datasets/kali-uint8-64x64.pt"),
                    torch.load(f"./datasets/kali-uint8-64x64-CLIP.pt"),
                ),
                dtype=torch.float, multiply=1. / 255.,
                # transforms=[VT.RandomCrop((64, 64))],
            ),
            TransformDataset(
                TensorDataset(
                    torch.load(f"./datasets/pattern-1x64x64-uint.pt"),
                    torch.load(f"./datasets/pattern-1x64x64-uint-CLIP.pt"),
                ),
                dtype=torch.float, multiply=1. / 255.,
                transforms=[lambda i: i.repeat(3, 1, 1)],
            ),
            TransformDataset(
                TensorDataset(
                    torch.load(f"./datasets/photos-64x64-bcr03.pt"),
                    torch.load(f"./datasets/photos-64x64-bcr03-CLIP.pt"),
                ),
            )
        ],
    )

    one_image, one_feature = ds[0]
    assert one_image.shape == SHAPE, one_image.shape
    assert one_image.dtype == torch.float, one_image.dtype
    assert one_feature.shape == (CODE_SIZE,)

    train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))
    print(f"{len(test_ds)} validation samples")

    #model = EncoderMLP(SHAPE, channels=[CODE_SIZE * 4, CODE_SIZE], hidden_act=nn.GELU())
    #model = EncoderConv(SHAPE, code_size=CODE_SIZE, channels=[64], kernel_size=32, pool_kernel_size=8, batch_norm=True)#, pool_type="average")
    model = EncoderTrans(SHAPE, code_size=CODE_SIZE)
    print(model)

    trainer = Trainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        # max_inputs=100_000_000_000,
        max_epoch=1000,
        data_loader=DataLoader(train_ds, batch_size=64, shuffle=True),
        validation_loader=DataLoader(test_ds, batch_size=50),
        # freeze_validation_set=True,
        optimizers=[
            #torch.optim.SGD(model.parameters(), lr=.001, weight_decay=0.00001),
            torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.00001),
            #torch.optim.Adadelta(model.parameters(), lr=.2),
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
