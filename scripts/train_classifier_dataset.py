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
import torchvision.models
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src import console
from src.train.trainer import Trainer
from src.models.cnn import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters


class AlexNet(torchvision.models.AlexNet):

    def weight_images(self, **kwargs):
        images = []

        for layer in (self.features[0], ):
            if hasattr(layer, "weight"):
                weight = layer.weight
                if weight.ndim == 4:
                    for wchan in weight[:64]:
                        for w in wchan[:3]:
                            images.append(w)
        return images


class VisionTransformer(torchvision.models.VisionTransformer):

    def weight_images(self, **kwargs):
        images = []

        for layer in (self.conv_proj, ):
            if hasattr(layer, "weight"):
                weight = layer.weight
                if weight.ndim == 4:
                    for wchan in weight[:64]:
                        for w in wchan[:3]:
                            images.append(w)
        return images


class ClassifierTrainer(Trainer):

    def train_step(self, input_batch) -> torch.Tensor:
        input, target_features, labels = input_batch
        output_features = self.model(input)

        class_loss = F.cross_entropy(output_features, target_features)
        #class_loss = (output_features, target_features)

        # std_loss = (.1 - output_features.std()).clamp_min(0)

        return class_loss# + std_loss

    def write_step(self):
        all_features = []
        num_inputs = 0
        num_correct = 0
        for input, target_features, labels in self.iter_validation_batches():
            output_features = self.model(input.to(self.device)).cpu()
            all_features.append(output_features)

            target_idx = torch.argmax(target_features, dim=-1, keepdim=True).reshape(-1)
            output_idx = torch.argmax(output_features, dim=-1, keepdim=True).reshape(-1)
            num_correct += (target_idx == output_idx).sum().item()
            num_inputs += input.shape[0]

        all_features = torch.cat(all_features)[:1000]
        # all_features = F.softmax(all_features, -1)

        self.log_image("validation_features", all_features.unsqueeze(0))
        self.log_scalar("validation_accuracy", num_correct / num_inputs * 100)
        self.log_scalar("validation_features_std", all_features.std())
        self.log_scalar("validation_features_mean", all_features.mean())


def main():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    SHAPE = (3, 112, 112)
    NUM_CLASSES = 1000
    ds = ConcatDataset(
        [
            ClassFeaturesDataset(
                TransformDataset(
                    TensorDataset(
                        torch.load(f"./datasets/ifs-1x128x128-uint8-1000x32.pt"),
                        torch.load(f"./datasets/ifs-1x128x128-uint8-1000x32-labels.pt")
                    ),
                    dtype=torch.float, multiply=1. / 255.,
                    transforms=[
                        #VT.RandomRotation(20),
                        VT.RandomCrop(SHAPE[-2:]),
                        lambda i: i.repeat(3, 1, 1),
                    ],
                    # num_repeat=5,
                ),
                tuple_position=1,
                off_value=.1,
            )
        ],
    )

    one_image, one_feature, _ = ds[0]
    assert one_image.shape == SHAPE, one_image.shape
    assert one_image.dtype == torch.float, one_image.dtype
    assert one_feature.shape == (NUM_CLASSES,), one_feature.shape

    num_valid = 2000
    num_train = len(ds) - num_valid
    train_ds, test_ds = torch.utils.data.random_split(ds, [num_train, num_valid], torch.Generator().manual_seed(42))
    print(f"{len(test_ds)} validation samples")

    #model = EncoderMLP(SHAPE, channels=[CODE_SIZE * 4, CODE_SIZE], hidden_act=nn.GELU())
    #model = ClassifierConv(SHAPE, num_classes=1000, code_size=CODE_SIZE, channels=[64], kernel_size=15, stride=5, pool_kernel_size=0, batch_norm=True)#, pool_type="average")
    #model = EncoderTrans(SHAPE, code_size=CODE_SIZE)
    model = AlexNet(num_classes=NUM_CLASSES)
    #model = VisionTransformer(
    #    image_size=SHAPE[-1], patch_size=28, num_layers=5, num_heads=4,
    #    mlp_dim=256, hidden_dim=128, dropout=.5, num_classes=NUM_CLASSES,
    #)
    print(model)

    trainer = ClassifierTrainer(
        **kwargs,
        model=model,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        # max_inputs=100_000_000_000,
        max_epoch=10000,
        data_loader=DataLoader(train_ds, batch_size=64, shuffle=True),
        validation_loader=DataLoader(test_ds, batch_size=512),
        freeze_validation_set=True,
        training_noise=0.1,
        optimizers=[
            #torch.optim.SGD(model.parameters(), lr=.005, weight_decay=0.00001),
            torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.),
            #torch.optim.Adadelta(model.parameters(), lr=1.),
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
