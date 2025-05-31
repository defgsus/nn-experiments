import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as VF
from jupyterlab.semver import valid
from networkx.algorithms.components import number_connected_components
from torchvision.utils import make_grid
import torchvision.datasets
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm

from src.datasets import Imagenet1kIterableDataset, WrapDataset
from src.models.util import activation_to_module
from src.util import to_torch_device, iter_parameter_permutations
from src.util.image import iter_image_patches, signed_to_image
from experiments.datasets import cifar10_dataset, stl10_dataset
from src import config as global_config


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "pca_conv"


@dataclasses.dataclass
class Config:
    pca_ds: Literal["stl", "imagenet"] = "stl"
    kernel_size: int = 7
    num_components: int = 64
    activation: Optional[str] = "relu"
    conv_stride: int = 2

    @property
    def slug(self) -> str:
        return f"pcads-{self.pca_ds}_ks{self.kernel_size}_nc{self.num_components}_s{self.conv_stride}_act-{self.activation}"


@torch.no_grad()
def train_pca(
        config: Config,
        cache_path: Path = CACHE_PATH,
        num_layers: int = 10,
        batch_size: int = 1000,
        device: str = "auto",
):
    device = to_torch_device(device)
    print("device:", device)

    cache_path = cache_path / config.slug

    if config.pca_ds == "imagenet":
        dataset = (
            Imagenet1kIterableDataset(
                size_filter=[(500, 500)],
            )
            #.shuffle(1000, seed=23)
            .limit(50)
        )
    else:
        dataset = (
            WrapDataset(torchvision.datasets.STL10(global_config.SMALL_DATASETS_PATH, split="train"))
            .transform([lambda x: VF.to_tensor(x.convert("RGB"))])
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
                    if isinstance(image, tuple):
                        image = image[0]

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


_CLASSIFICATION_DATASET = None
def get_classification_dataset():
    global _CLASSIFICATION_DATASET
    if _CLASSIFICATION_DATASET is None:
        print("loading dataset")
        ds = torchvision.datasets.STL10(global_config.SMALL_DATASETS_PATH, split="train")
        train_x, train_y = ds.data.astype(np.float32) / 255., ds.labels
        ds = torchvision.datasets.STL10(global_config.SMALL_DATASETS_PATH, split="test")
        test_x, test_y = ds.data.astype(np.float32) / 255., ds.labels

        _CLASSIFICATION_DATASET = train_x, train_y, test_x, test_y

    return _CLASSIFICATION_DATASET


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

    train_x, train_y, test_x, test_y = get_classification_dataset()

    train_x = _process(train_x)
    test_x = _process(test_x)

    classifier = RidgeClassifier()
    print("fitting classifier")
    start_time = time.time()
    classifier.fit(train_x, train_y)
    train_time = time.time() - start_time
    print("predicting validation set")
    predicted_train_y = classifier.predict(train_x)
    print("predicting validation set")
    predicted_val_y = classifier.predict(test_x)

    train_accuracy = (predicted_train_y == train_y).astype(np.float32).mean() * 100
    val_accuracy = (predicted_val_y == test_y).astype(np.float32).mean() * 100

    print(f"classification accuracy l{max_layers}-{config.slug}{'(random)' if random_weights else ''}: {val_accuracy}, {train_accuracy}%")

    return {
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "train_time": train_time,
    }


def test_random_convs(num_repeats: int = 5):
    param_matrix = {
        "kernel_size": [3, 5, 7, 9, 11],
        "num_components": [32, 64],
        "activation": ["relu"],
        "max_layers": [1, 2, 3, 4],
        "conv_stride": [2],
    }
    for params in iter_parameter_permutations(param_matrix):
        max_layers = params.pop("max_layers")
        config = Config(**params)

        filename = CACHE_PATH / "random" / f"l{max_layers}_{config.slug}.json"

        if filename.exists():
            continue

        results = []
        for repeat_idx in range(num_repeats):
            try:
                result = test_classification(
                    config=config,
                    max_layers=max_layers,
                    random_weights=True,
                )
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                continue

            results.append(result)

        os.makedirs(filename.parent, exist_ok=True)
        filename.write_text(json.dumps({
            "parameters": {**params, "num_layers": max_layers},
            "results": results,
        }))


def print_random_convs_table():
    path = CACHE_PATH / "random"
    rows = []
    for file in path.glob("*.json"):
        data = json.loads(file.read_text())
        data["parameters"].setdefault("conv_stride", 2)

        if not data["results"]:
            continue

        row = data["parameters"]
        rows.append(row)

        row.update({
            "train_accuracy": sum(r["train_accuracy"] for r in data["results"]) / len(data["results"]),
            "val_accuracy": sum(r["val_accuracy"] for r in data["results"]) / len(data["results"]),
        })

    df = pd.DataFrame(rows).sort_values("val_accuracy")

    print(df.to_markdown())


if __name__ == "__main__":

    config = Config(
        kernel_size=7,
        num_components=64,
        activation="relu",
    )

    #train_pca(config=config, num_layers=6)
    #test_classification(config=config, random_weights=False, max_layers=2)

    test_random_convs()
    print_random_convs_table()
