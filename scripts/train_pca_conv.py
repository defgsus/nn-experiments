import dataclasses
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as VF
from networkx.algorithms.components import number_connected_components
from torchvision.utils import make_grid
import torchvision.datasets
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm

from src.datasets import Imagenet1kIterableDataset
from src.models.util import activation_to_module
from src.util import to_torch_device
from src.util.image import iter_image_patches, signed_to_image
from experiments.datasets import cifar10_dataset, stl10_dataset


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "pca_conv"


@dataclasses.dataclass
class Config:
    kernel_size: int = 7
    num_components: int = 64
    activation: Optional[str] = "relu"
    conv_stride: int = 2

    @property
    def slug(self) -> str:
        return f"ks{self.kernel_size}-nc{self.num_components}-s{self.conv_stride}-{self.activation}"


@torch.no_grad()
def train(
        config: Config,
        cache_path: Path = CACHE_PATH,
        num_layers: int = 10,
        batch_size: int = 1000,
        device: str = "auto",
):
    device = to_torch_device(device)
    print("device:", device)

    cache_path = cache_path / config.slug

    dataset = (
        Imagenet1kIterableDataset(
            size_filter=[(500, 500)],
        )
        #.shuffle(1000, seed=23)
        .limit(50)
    )

    conv_layers = nn.Sequential().to(device)
    num_input_components = 3
    patch_stride = config.kernel_size // 2

    for layer_index in range(num_layers):

        if layer_index > 2:
            patch_stride = max(1, patch_stride - 1)

        print(f"layer#{layer_index:02}: patch_stride={patch_stride}")

        cache_filename = cache_path / f"layer-{layer_index:02}.pt"
        if not cache_filename.exists():

            pca = IncrementalPCA(config.num_components)

            def _iter_patches():
                output_batch = []
                demo_images = []
                for image in tqdm(dataset, desc=f"layer#{layer_index:02}"):

                    # print("IMAGE IN: ", image.shape)
                    image = conv_layers(image.to(device)).cpu()
                    # print("IMAGE OUT:", image.shape)
                    #if any(s < kernel_size for s in image.shape[-2:]):
                    #    continue
                    if len(demo_images) < 4:
                        demo_images.append(image[:3].clamp(0, 1))

                    for patch in iter_image_patches(
                            image=image,
                            shape=(config.kernel_size, config.kernel_size),
                            stride=patch_stride,
                    ):
                        output_batch.append(patch.unsqueeze(0))

                        if len(output_batch) >= batch_size:
                            yield torch.concat(output_batch)
                            output_batch.clear()

                if output_batch:
                    yield torch.concat(output_batch)

                os.makedirs(cache_filename.parent, exist_ok=True)
                VF.to_pil_image(
                    make_grid(demo_images)
                ).save(cache_filename.parent / f"layer-{layer_index:02}-images.png")

            for patch_batch in _iter_patches():
                pca.partial_fit(
                    patch_batch.numpy().reshape(
                        patch_batch.shape[0],
                        num_input_components * config.kernel_size ** 2,
                    )
                )

            weights = torch.from_numpy(pca.components_).reshape(
                config.num_components, num_input_components, config.kernel_size, config.kernel_size,
            )
            os.makedirs(cache_filename.parent, exist_ok=True)
            torch.save(weights, cache_filename)

            VF.to_pil_image(
                make_grid([normalize(p[:3]) for p in weights], nrow=16)
            ).save(cache_filename.with_suffix(".png"))

            VF.to_pil_image(
                make_grid([
                    signed_to_image(p0)
                    for p in weights
                    for p0 in p
                ], nrow=num_input_components)
            ).save(cache_filename.parent / f"layer-{layer_index:02}-all.png")

        else:
            weights = torch.load(cache_filename)

        conv = nn.Conv2d(num_input_components, config.num_components, config.kernel_size, stride=config.conv_stride, bias=False)
        conv.weight[:] = weights
        conv_layers.append(conv)
        if config.activation is not None:
            conv_layers.append(activation_to_module(config.activation))
        conv_layers.to(device)

        num_input_components = config.num_components


def normalize(patch: torch.Tensor):
    patch = patch - patch.min()
    patch = patch / patch.max()
    return patch.clamp(0, 1)


def load_layers(
        config: Config,
        cache_path: Path = CACHE_PATH,
        max_layers: Optional[int] = None,
):
    files = sorted((cache_path / config.slug).glob("layer-*.pt"))
    if max_layers is not None:
        files = files[:max_layers]
    if not files:
        raise ValueError(f"No data for config slug '{config.slug}'")

    layers = nn.Sequential()
    num_input_components = 3
    for file in files:
        weight = torch.load(file)
        conv = nn.Conv2d(num_input_components, config.num_components, config.kernel_size, stride=config.conv_stride, bias=False)
        conv.weight[:] = weight
        layers.append(conv)
        if config.activation is not None:
            layers.append(activation_to_module(config.activation))
        num_input_components = config.num_components

    return layers


def create_random_layers(
        config: Config,
        num_layers: int,
):
    layers = nn.Sequential()
    num_input_components = 3
    for i in range(num_layers):
        conv = nn.Conv2d(
            num_input_components, config.num_components, config.kernel_size, stride=config.conv_stride,
            bias=False,
        )
        layers.append(conv)
        if config.activation is not None:
            layers.append(activation_to_module(config.activation))
        num_input_components = config.num_components

    return layers


@torch.no_grad()
def test_classification(
        config: Config,
        cache_path: Path = CACHE_PATH,
        # since the STL10 is pretty small (96²), need to limit layers when using conv_stride
        max_layers: int = 1,
        random_weights: bool = False,
):
    """
    Check classification accuracy by linear probing.

    Accuracy for STL10 with RidgeClassifier:

        raw images:                     21.8125%
        l3-ks7-nc64-s2-tanh:            23.3625%
        l1-ks7-nc64-s2-sigmoid:         23.8250%
        l1-ks7-nc64-s2-tanh(random):    23.9000%
        l1-ks7-nc64-s2-sigmoid(random): 24.7125%
        l3-ks7-nc64-s2-None:            25.4250%
        l2-ks7-nc64-s2-tanh(random):    25.5875%
        l3-ks7-nc64-s2-sigmoid(random): 27.6000%
        l3-ks7-nc64-s2-None(random):    28.6875%
        l3-ks7-nc64-s2-tanh(random):    30.5750%
        l3-ks7-nc64-s2-relu:            32.6000%
        l2-ks7-nc64-s2-sigmoid:         34.1375%
        l2-ks7-nc64-s2-sigmoid(random): 35.3875%
        l3-ks7-nc64-s2-relu(random):    41.2500%
        l2-ks7-nc64-s2-relu:            42.6875%
        l1-ks7-nc64-s2-relu:            44.1500%
    """
    if not random_weights:
        layers = load_layers(
            config=config,
            cache_path=cache_path,
            max_layers=max_layers,
        )
    else:
        layers = create_random_layers(config, num_layers=max_layers)
    print(layers)

    def _process(images: np.ndarray, batch_size: int = 32):
        result = []
        with tqdm(total=images.shape[0], desc="processing images") as progress:
            for i in range((images.shape[0] + batch_size - 1) // batch_size):
                batch = torch.from_numpy(images[i * batch_size: (i + 1) * batch_size])
                batch = layers(batch)
                result.append(batch.reshape(batch.shape[0], -1).numpy())
                progress.update(batch.shape[0])
        return np.concat(result, axis=0)

    print("loading dataset")
    ds = torchvision.datasets.STL10("~/prog/data/datasets/", split="train")
    train_x, train_y = ds.data.astype(np.float32) / 255., ds.labels
    ds = torchvision.datasets.STL10("~/prog/data/datasets/", split="test")
    test_x, test_y = ds.data.astype(np.float32) / 255., ds.labels

    train_x = _process(train_x)
    test_x = _process(test_x)

    classifier = RidgeClassifier()
    print("fitting classifier")
    classifier.fit(train_x, train_y)
    print("predicting validation set")
    predicted_y = classifier.predict(test_x)

    accuracy = (predicted_y == test_y).astype(np.float32).mean() * 100

    print(f"classification accuracy l{max_layers}-{config.slug}{'(random)' if random_weights else ''}: {accuracy}%")


if __name__ == "__main__":

    config = Config(
        kernel_size=7,
        num_components=64,
        activation="sigmoid",
    )

    train(config=config, num_layers=6)
    test_classification(config=config, random_weights=False, max_layers=1)
