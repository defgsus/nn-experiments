import math
import argparse
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Generator

import numpy as np
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
from sklearn.cluster import MiniBatchKMeans

from src import console
from src.train import Trainer
from src.models.cnn import *
from src.models.generative import *
from src.datasets import *
from src.util.image import *
from src.util import num_module_parameters
from src.algo import Space2d
from src.models.vae import *

from scripts.train_classifier_dataset import AlexNet


class TrainUnsupervisedClustering(Trainer):

    def __init__(self, *args, n_clusters: int = 100, **kwargs):
        if kwargs.get("freeze_validation_set"):
            raise NotImplementedError("Sorry, `freeze_validation_set` can currently not be used")
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self._clusterer = None
        self._cluster_labels: Optional[torch.Tensor] = None
        self._cluster_labels_v: Optional[torch.Tensor] = None
        self._cluster_label_features: Optional[torch.Tensor] = None
        self._cluster_label_features_v: Optional[torch.Tensor] = None

    def every_3_epoch(self):
        self.reset_cluster()

    def reset_cluster(self):
        self._cluster_labels = None
        self._cluster_labels_v = None
        self._cluster_label_features = None
        self._cluster_label_features_v = None

    def train_step(self, input_batch) -> torch.Tensor:
        input, target_features = input_batch
        output_features = self.model(input)
        return F.cross_entropy(output_features, target_features)

    def iter_training_batches(self) -> Generator:
        """
        Yields (training_batch, target_features)
            where target_features are the class-labels as desired classifier output
        """
        bs = self.data_loader.batch_size
        for batch_idx, batch in enumerate(self.data_loader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            self._update_cluster_labels()
            yield batch, self._cluster_label_features[batch_idx * bs: (batch_idx + 1) * bs]

    def iter_validation_batches(self) -> Generator:
        bs = self.validation_loader.batch_size
        for batch_idx, batch in enumerate(self.validation_loader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            self._update_cluster_labels()
            yield batch, self._cluster_label_features_v[batch_idx * bs: (batch_idx + 1) * bs]

    def _update_cluster_labels(self):
        if self._cluster_labels is None:
            dtype, self._cluster_labels, self._cluster_labels_v = self.get_cluster_labels()

            self._cluster_label_features = np.zeros(
                (self._cluster_labels.shape[0], self.n_clusters),
            )
            for i, label in enumerate(self._cluster_labels):
                self._cluster_label_features[i][int(label)] = 1
            self._cluster_label_features = torch.Tensor(self._cluster_label_features).to(dtype)

            self._cluster_label_features_v = np.zeros(
                (self._cluster_labels_v.shape[0], self.n_clusters),
            )
            for i, label in enumerate(self._cluster_labels_v):
                self._cluster_label_features_v[i][int(label)] = 1
            self._cluster_label_features_v = torch.Tensor(self._cluster_label_features_v).to(dtype)

    @torch.no_grad()
    def get_cluster_labels(self, batch_size: int = 1024):
        if self._clusterer is None:
            self._clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=batch_size,
                random_state=23,
                n_init=3,
            )
        dtype = None

        def _fit(what, data_loader):
            nonlocal dtype

            dl = DataLoader(data_loader.dataset, batch_size=batch_size)
            for batch in tqdm(dl, desc=f"clustering {what}"):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                features = self.model.encoder(batch.to(self.device))
                dtype = features.dtype

                self._clusterer.partial_fit(features.cpu().numpy())

        _fit("training-data", self.data_loader)
        _fit("validation-data", self.validation_loader)

        def _get_labels(what, data_loader):
            labels = []
            for batch in tqdm(data_loader, desc=f"get {what} labels"):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                features = self.model.encoder(batch.to(self.device)).cpu().numpy()

                labels.append(self._clusterer.predict(features))
            return np.concatenate(labels)

        labels_train = _get_labels("training-data", self.data_loader)
        labels_validation = _get_labels("validation-data", self.validation_loader)
        return dtype, labels_train, labels_validation

    def write_step(self):
        all_features = []
        num_inputs = 0
        num_correct = 0
        feature_count = 0
        for input, target_features in self.iter_validation_batches():
            output_features = self.model(input.to(self.device)).cpu()

            if feature_count < 1000:
                all_features.append(output_features)
                feature_count += output_features.shape[0]

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


class ConvEncoder(nn.Module):

    def __init__(
            self,
            channels: Iterable[int],
            shape: Tuple[int, int, int],
            code_size: int,
            n_classes: int,
            kernel_size: int = 5,
            stride: int = 1,
    ):
        super().__init__()
        self.channels = list(channels)
        self.shape = shape
        encoder_block = Conv2dBlock(channels=self.channels, kernel_size=kernel_size, stride=stride, act_fn=nn.GELU())
        conv_shape = encoder_block.get_output_shape(self.shape)
        self.encoder = torch.nn.Sequential(
            encoder_block,
            nn.Flatten(),
            nn.Linear(math.prod(conv_shape), code_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(code_size, n_classes),
            #nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def weight_images(self, **kwargs):
        return self.encoder[0].weight_images(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    TrainUnsupervisedClustering.add_parser_args(parser)
    kwargs = vars(parser.parse_args())

    if 1:
        SHAPE = (1, 64, 64)
        ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{SHAPE[-2]}x{SHAPE[-1]}.pt")[:])
        #ds = TensorDataset(torch.load(f"./datasets/kali-uint8-{128}x{128}.pt"))
        ds = TransformDataset(
            ds,
            dtype=torch.float, multiply=1. / 255.,
            transforms=[
                #VT.CenterCrop(64),
                #VT.RandomCrop(SHAPE[-2:]),
                VT.Grayscale(),
            ],
            num_repeat=1,
        )
    else:
        SHAPE = (1, 64, 64)
        ds = TensorDataset(torch.load(f"./datasets/pattern-{SHAPE[-3]}x{SHAPE[-2]}x{SHAPE[-1]}-uint.pt")[:10])
        ds = TransformDataset(
            ds,
            dtype=torch.float, multiply=1. / 255.,
            num_repeat=2000,
        )
        assert ds[0][0].shape[:3] == torch.Size(SHAPE), ds[0][0].shape

    num_valid = 2000
    num_train = len(ds) - num_valid
    train_ds, test_ds = torch.utils.data.random_split(ds, [num_train, num_valid], torch.Generator().manual_seed(42))
    print(f"{len(test_ds)} validation samples")

    #train_ds = FontDataset(shape=SHAPE)
    #test_ds = TensorDataset(torch.load("./datasets/fonts-32x32.pt")[:500])

    NUM_CLUSTERS = 512
    CODE_SIZE = 512
    # model = AlexNet(CODE_SIZE)
    model = ConvEncoder(channels=[SHAPE[0], 32], shape=SHAPE, code_size=CODE_SIZE, n_classes=NUM_CLUSTERS, kernel_size=7, stride=2)
    print(model)

    trainer = TrainUnsupervisedClustering(
        **kwargs,
        model=model,
        n_clusters=NUM_CLUSTERS,
        #min_loss=0.001,
        num_epochs_between_validations=1,
        #num_inputs_between_validations=10_000,
        data_loader=DataLoader(train_ds, batch_size=64, shuffle=False),
        validation_loader=DataLoader(test_ds, batch_size=64),
        freeze_validation_set=False,
        optimizers=[
            torch.optim.Adam(model.parameters(), lr=.001),#, weight_decay=0.00001),
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
