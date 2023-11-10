import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict, Type

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
from src.train.train_autoencoder import TrainAutoencoder
from src.models.cnn import *
from src.models.generative import *
from src.models.transform import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters
from src.algo import Space2d

from scripts.train_classifier_dataset import AlexNet
from scripts import datasets


class Encoder(nn.Module):

    def __init__(self, channels: Iterable[int], shape: Tuple[int, int, int], code_size: int = 100):
        super().__init__()
        self.channels = list(channels)
        self.shape = shape
        encoder_block = Conv2dBlock(channels=self.channels, kernel_size=5, act_fn=nn.GELU())
        conv_shape = (self.channels[-1], *encoder_block.get_output_shape(self.shape[-2:]))
        self.layers = torch.nn.Sequential(
            encoder_block,
            nn.Flatten(),
            nn.Linear(math.prod(conv_shape), code_size),
        )

    def forward(self, x):
        return self.layers(x)


class Sinus(nn.Module):
    def __init__(self, size: int, freq_scale: float = 3.):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(size) * freq_scale)
        self.phase = nn.Parameter(torch.randn(size) * 3.)

    def forward(self, x):
        return torch.sin(x * self.freq + self.phase)


class Decoder(FreescaleImageModule):

    def __init__(self, code_size: int = 100):
        super().__init__(num_in=code_size)
        self.layers = nn.Sequential(
            nn.Linear(code_size + 2, code_size),
            Sinus(code_size, 30),
            nn.Linear(code_size, code_size),
            Sinus(code_size, 20),
            nn.Linear(code_size, 3),
            nn.Linear(3, 3),

        )

    def forward_state(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        return self.layers(x)


class NewAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.encoder = Encoder([shape[0], 20, 20], shape=shape)
        self.decoder = Decoder()
        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x), shape=self.shape)


class TransformerAutoencoder(ConvAutoEncoder):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: Iterable[int] = None,
            kernel_size: int = 5,
            code_size: int = 128,
            patch_size: int = 32,
            act_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
            batch_norm: bool = False,
    ):
        assert shape[-2] == shape[-1], shape

        super(TransformerAutoencoder, self).__init__(
            shape=shape, channels=channels, kernel_size=kernel_size, code_size=code_size,
            act_fn=act_fn, batch_norm=batch_norm,
        )
        self.encoder = VisionTransformer(
            shape[-1], patch_size=patch_size, width=256, layers=10, heads=8, output_dim=code_size
        )

    def weight_images(self, **kwargs):
        pass


class AlexAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], code_size: int):
        from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
        super().__init__()
        self.shape = shape
        self.encoder = AlexNet(num_classes=code_size)
        #self.encoder = ResNet(
        #    block=Bottleneck,
        #    layers=[2, 2, 2, 2], num_classes=code_size,
        #)
        self.decoder = nn.Sequential(
            nn.Linear(code_size, code_size * 2),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(code_size * 2, math.prod(shape)),
        )
        self.decoder = Decoder(code_size)
        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x), shape=self.shape)#.reshape(-1, *self.shape)

    def weight_images(self, **kwargs):
        pass#return self.encoder.weight_images(**kwargs)


class MLPAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], channels: Iterable[int]):
        super().__init__()
        self.channels = [math.prod(shape), *channels]
        self.shape = shape
        self.code_size = self.channels[-1]
        self.encoder = nn.Sequential(
            nn.Flatten(),
        )
        for i, (chan, next_chan) in enumerate(zip(self.channels, self.channels[1:])):
            self.encoder.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.encoder.append(nn.ReLU())
                # self.encoder.append(nn.Dropout())

        self.decoder = nn.Sequential()
        channels = list(reversed(self.channels))
        for i, (chan, next_chan) in enumerate(zip(channels, channels[1:])):
            self.decoder.append(nn.Linear(chan, next_chan))
            if i < len(self.channels) - 2:
                self.decoder.append(nn.ReLU())
                # self.decoder.append(nn.Dropout())
        self.decoder.append(Reshape(self.shape))


        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def weight_images(self, **kwargs):
        images = []
        for w in self.encoder[1].weight.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        for w in self.decoder[-2].weight.T.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        return images


class MLPDetailAutoEncoder(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], code_size: int):
        super().__init__()
        self.shape = shape
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(shape), code_size),
        )

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.coarse = nn.Linear(code_size, math.prod(shape))
                self.fine = nn.Sequential(
                    nn.Linear(math.prod(shape) + code_size, math.prod(shape))
                )
            def forward(self, x):
                y = self.coarse(x)
                y2 = self.fine(torch.cat([y, x], dim=-1))
                y = (y + y2).reshape(-1, *shape)
                return y

        self.decoder = Decoder()

        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)).reshape(-1, *self.shape)

    def weight_images(self, **kwargs):
        images = []
        for w in self.encoder[-1].weight.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        for w in self.decoder.coarse.weight.T.reshape(-1, *self.shape)[:32]:
            for w1 in w:
                images.append(w1)
        return images


class DalleAutoencoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            n_hid: int = 256,
            vocab_size: int = 128,
            group_count: int = 4,
            n_blk_per_group: int = 2,
            use_mixed_precision: bool = False,
            act_fn: Type[nn.Module] = nn.ReLU,
            space_to_depth: bool = False,
    ):
        from src.models.cnn import DalleEncoder, DalleDecoder
        super().__init__()
        self.shape = shape
        encoder = DalleEncoder(
            n_hid=n_hid, requires_grad=True, vocab_size=vocab_size,
            input_channels=self.shape[0], use_mixed_precision=use_mixed_precision,
            group_count=group_count, n_blk_per_group=n_blk_per_group,
            act_fn=act_fn,
            space_to_depth=space_to_depth,
        )
        with torch.no_grad():
            out_shape = encoder(torch.zeros(1, *self.shape)).shape
        self.encoder = nn.Sequential(
            encoder,
            nn.Flatten(1),
            nn.Linear(math.prod(out_shape), vocab_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(vocab_size, math.prod(out_shape)),
            Reshape(out_shape[-3:]),
            DalleDecoder(
                n_hid=n_hid, vocab_size=vocab_size, requires_grad=True,
                output_channels=self.shape[0], use_mixed_precision=use_mixed_precision,
                group_count=group_count,
                n_blk_per_group=n_blk_per_group,
                act_fn=act_fn,
            ),
        )
        print(f"encoder params: {num_module_parameters(self.encoder):,}")
        print(f"decoder params: {num_module_parameters(self.decoder):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DalleManifoldAutoencoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            n_hid: int = 256,
            vocab_size: int = 128,
            group_count: int = 4,
            n_blk_per_group: int = 2,
            use_mixed_precision: bool = False,
            act_fn: Type[nn.Module] = nn.ReLU,
            space_to_depth: bool = False,
            decoder_n_hid: int = 256,
            decoder_n_blk: int = 2,
            decoder_n_layer: int = 2,
            decoder_concat_residual: Union[bool, Iterable[bool]] = False,
    ):
        from src.models.cnn import DalleEncoder, DalleDecoder
        from src.models.decoder.image_manifold import ImageManifoldDecoder

        super().__init__()
        self.shape = shape
        encoder = DalleEncoder(
            n_hid=n_hid, requires_grad=True, vocab_size=vocab_size,
            input_channels=self.shape[0], use_mixed_precision=use_mixed_precision,
            group_count=group_count, n_blk_per_group=n_blk_per_group,
            act_fn=act_fn,
            space_to_depth=space_to_depth,
        )
        with torch.no_grad():
            out_shape = encoder(torch.zeros(1, *self.shape)).shape
        self.encoder = nn.Sequential(
            encoder,
            nn.Flatten(1),
            nn.Linear(math.prod(out_shape), vocab_size)
        )
        self.decoder = ImageManifoldDecoder(
            num_input_channels=vocab_size,
            num_output_channels=shape[0],
            num_hidden=decoder_n_hid,
            num_blocks=decoder_n_blk,
            num_layers_per_block=decoder_n_layer,
            default_shape=shape[-2:],
            concat_residual=decoder_concat_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def weight_images(self, **kwargs):
        return self.decoder.weight_images(**kwargs)


class ManifoldAutoencoder(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            vocab_size: int = 128,
            n_hid: int = 256,
    ):
        from src.models.decoder.image_manifold import ImageManifoldDecoder, ImageManifoldEncoder

        super().__init__()
        self.shape = shape
        self.encoder = ImageManifoldEncoder(
            num_input_channels=shape[0],
            num_output_channels=vocab_size,
            num_hidden=n_hid,
            # num_blocks=
            # num_layers_per_block=
        )
        self.decoder = ImageManifoldDecoder(
            num_input_channels=vocab_size,
            num_output_channels=shape[0],
            num_hidden=n_hid,
            # num_blocks=
            # num_layers_per_block=
            default_shape=shape[-2:],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def weight_images(self, **kwargs):
        return self.decoder.weight_images(**kwargs)


class Sobel(nn.Module):
    def __init__(self, kernel_size: int = 5, sigma: float = 5.):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        blurred = VF.gaussian_blur(x, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])
        return (x - blurred).clamp_min(0)


class TrainAutoencoderSpecial(TrainAutoencoder):

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        input_batch = input_batch[0]
        if hasattr(self.model, "encode"):
            feature_batch = self.model.encode(input_batch)
        else:
            feature_batch = self.model.encoder(input_batch)
        if hasattr(self.model, "decode"):
            output_batch = self.model.decode(feature_batch)
        else:
            output_batch = self.model.decoder(feature_batch)

        if input_batch.shape != output_batch.shape:
            raise ValueError(
                f"input_batch = {input_batch.shape}"
                f", output_batch = {output_batch.shape}"
                f", feature_batch = {feature_batch.shape}"
            )

        reconstruction_loss = self.loss_function(input_batch, output_batch)

        if 0:
            if not hasattr(self, "_sobel_filter"):
                self._sobel_filter = Sobel()

            sobel_input_batch = self._sobel_filter(input_batch)
            sobel_output_batch = self._sobel_filter(output_batch)
            sobel_reconstruction_loss = self.loss_function(sobel_input_batch, sobel_output_batch)

        loss_batch_std = (.5 - feature_batch.std(0).mean()).abs()
        loss_batch_mean = (0. - feature_batch.mean()).abs()

        return {
            "loss": (
                reconstruction_loss
                + .0001 * (loss_batch_std + loss_batch_mean)
                # + 1. * sobel_reconstruction_loss
            ),
            "loss_reconstruction": reconstruction_loss,
            #"loss_reconstruction_sobel": sobel_reconstruction_loss,
            "loss_batch_std": loss_batch_std,
            "loss_batch_mean": loss_batch_mean,
        }


def create_kali_rpg_dataset(shape: Tuple[int, int, int]) -> IterableDataset:
    datasets_path = Path(__file__).resolve().parent.parent / "datasets"

    ds_kali = TensorDataset(torch.load(datasets_path / f"kali-uint8-{128}x{128}.pt"))
    ds_kali = TransformDataset(
        ds_kali,
        dtype=torch.float, multiply=1. / 255.,
        transforms=[
            #VT.CenterCrop(64),
            VT.RandomCrop(shape[-2:]),
        ],
        num_repeat=1,
    )

    ds_rpg = datasets.RpgTileIterableDataset(shape)
    ds_rpg = TransformIterableDataset(
        ds_rpg,
        transforms=[
            VT.Pad(3),
            VT.RandomCrop(shape[-2:]),
            VT.RandomHorizontalFlip(.4),
            VT.RandomVerticalFlip(.2),
        ],
        num_repeat=2,
    )

    ds = InterleaveIterableDataset(
        [ds_kali, ds_rpg],
    )
    def _print(x):
        print("X", x.shape)
        if x.shape != torch.Size((1, 32, 32)):
            print("XXX", x.shape)
        return x

    ds = TransformIterableDataset(
        ds, transforms=[
            lambda x: set_image_channels(x, shape[0]),
            #lambda x: x[0] if isinstance(x, (list, tuple)) else x,
            #_print,
        ],
        remove_tuple=True,
    )
    return IterableShuffle(ds, max_shuffle=5000)


def create_rpg_dataset(shape: Tuple[int, int, int]) -> IterableDataset:
    ds_rpg = datasets.RpgTileIterableDataset(shape)
    ds_rpg = TransformIterableDataset(
        ds_rpg,
        transforms=[
            lambda x: set_image_channels(x, shape[0]),
            #VT.Pad(3),
            #VT.RandomCrop(shape[-2:]),
            VT.RandomHorizontalFlip(.4),
            VT.RandomVerticalFlip(.2),
        ],
        num_repeat=10,
    )
    ds_rpg = LimitIterableDataset(ds_rpg, 300 * 10)
    return ds_rpg


def main():
    parser = argparse.ArgumentParser()
    TrainAutoencoder.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (1, 32, 32)
    #ds = create_kali_rpg_dataset(SHAPE)
    ds = create_rpg_dataset(SHAPE)

    if isinstance(ds, IterableDataset):
        train_ds = ds
        # NOT a true validation set!
        test_ds = TensorDataset(torch.concat([
            x[0].unsqueeze(0) if isinstance(x, (tuple, list)) else x.unsqueeze(0)
            for i, x in zip(range(1000), ds)
        ]))
    else:
        train_ds, test_ds = torch.utils.data.random_split(ds, [0.99, 0.01], torch.Generator().manual_seed(42))

    sample = next(iter(train_ds))
    assert sample.shape[:3] == torch.Size(SHAPE), sample.shape
    sample = next(iter(test_ds))
    assert sample[0].shape[:3] == torch.Size(SHAPE), sample[0].shape

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    #model = NewAutoEncoder(SHAPE)
    #model = ConvAutoEncoder(SHAPE, channels=[8, 16, 24], kernel_size=7, code_size=128) # good one
    #model = ConvAutoEncoder(SHAPE, channels=[8, 16, 24], kernel_size=7, code_size=128, bias=False, linear_bias=False, act_last_layer=False)
    #model = TransformerAutoencoder(SHAPE, channels=[8, 16, 24], kernel_size=7, code_size=64)
    #model = ConvAutoEncoder(SHAPE, channels=[32], kernel_size=7, code_size=64)
    #model = MLPAutoEncoder(SHAPE, [128])
    #model = MLPDetailAutoEncoder(SHAPE, 128)
    #model = VariationalAutoencoder(SHAPE, 128)

    #model = ConvAutoEncoder(SHAPE, channels=[32, 32, 32], kernel_size=[3, 6, 8], code_size=128, space_to_depth=True)
    #model = ConvAutoEncoder(SHAPE, channels=[16, 16, 16], kernel_size=[3, 6, 4], code_size=128, space_to_depth=True)
    #model = ConvAutoEncoder(SHAPE, channels=[16,], kernel_size=[3,], code_size=32, space_to_depth=True)

    # val: 0.0099 (200k), 0.0091 (1M)
    #model = DalleAutoencoder(SHAPE, vocab_size=128, n_hid=64)

    # ae-d3: val: 0.01 (200k), 0.00889 (2M), 0.00857 (5M)
    #model = DalleAutoencoder(SHAPE, vocab_size=128, n_hid=64, group_count=1, n_blk_per_group=1, act_fn=nn.GELU)
    #model = DalleAutoencoder(SHAPE, vocab_size=128, n_hid=64, group_count=1, n_blk_per_group=1, act_fn=nn.GELU, space_to_depth=True)
    #model = DalleAutoencoder(SHAPE, vocab_size=128, n_hid=96, group_count=4, n_blk_per_group=2, act_fn=nn.GELU, space_to_depth=True)
    #model = DalleManifoldAutoencoder(SHAPE, vocab_size=128, n_hid=64, n_blk_per_group=1, act_fn=nn.GELU, space_to_depth=True)
    # ae-manifold-7
    #model = DalleManifoldAutoencoder(SHAPE, vocab_size=128, n_hid=64, n_blk_per_group=1, act_fn=nn.GELU, space_to_depth=True, decoder_n_blk=8, decoder_n_layer=2, decoder_n_hid=256)
    # ae-manifold-8
    #model = DalleManifoldAutoencoder(SHAPE, vocab_size=128, n_hid=64, n_blk_per_group=2, act_fn=nn.GELU, space_to_depth=True, decoder_n_blk=8, decoder_n_layer=2, decoder_n_hid=300)
    model = DalleManifoldAutoencoder(SHAPE, vocab_size=128, n_hid=64, n_blk_per_group=1, act_fn=nn.GELU, space_to_depth=True, decoder_n_blk=8, decoder_n_layer=2, decoder_n_hid=64, decoder_concat_residual=[True, False] * 4)
    state = torch.load("./checkpoints/ae-manifold-7/best.pt")
    model.encoder.load_state_dict({key[8:]: value for key, value in state["state_dict"].items() if key.startswith("encoder.")})
    #model = ManifoldAutoencoder(SHAPE, vocab_size=128, n_hid=256)
    print(model)
    for key in ("encoder", "decoder"):
        if hasattr(model, key):
            print(f"{key} params: {num_module_parameters(getattr(model, key)):,}")

    optimizer = (
        #torch.optim.Adam(model.parameters(), lr=.0002)#, weight_decay=0.00001)
        torch.optim.AdamW(model.parameters(), lr=.0003)#, weight_decay=0.00001)
        #torch.optim.Adadelta(model.parameters(), lr=1.)
    )

    BATCH_SIZE = 64
    MAX_INPUTS = 1_000_000  # 10_000_000
    print("MAX_INPUTS", MAX_INPUTS)

    trainer = TrainAutoencoderSpecial(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        max_inputs=MAX_INPUTS,
        data_loader=DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=not isinstance(train_ds, IterableDataset)),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=True,
        loss_function="l2",
        optimizers=[optimizer],
        schedulers=[
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_INPUTS // BATCH_SIZE)
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
