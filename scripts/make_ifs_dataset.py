"""
Generates a multi-class IteratedFunctionSystem image dataset with variations per class

```shell
# generate (in parallel)
# creates 5 datasets with 200 classes x 32 variations each
python scripts/make_ifs_dataset.py -nc 200 -nv 32 -s 100_000 200_000 300_000 400_000 500_000

# combine to one dataset
python scripts/make_ifs_dataset.py -nc 200 -nv 32 -m combine
```
"""
import argparse
import re
import json
import math
import random
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Iterable, Generator
from multiprocessing import Pool

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

import numpy as np
import pandas as pd

from src.datasets import *
from src.util.image import *
from src.util import ImageFilter
from src.algo import IFS
from src.datasets.generative import IFSClassIterableDataset


def get_answer(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} (Y/n)? ")
        if answer in ("n", "N"):
            return False
        if answer in ("", "y", "Y"):
            return True


def store_dataset(
        images: Iterable,
        output_filename: str,
        max_megabyte: int = 4096,
        tqdm_label: Optional[str] = None,
        tqdm_position: int = 0,
):
    tensor_batch = []
    label_batch = []
    tensor_size = 0
    last_print_size = 0
    try:
        for image, label in tqdm(images, desc=tqdm_label, position=tqdm_position):

            image = (image.clamp(0, 1) * 255).to(torch.uint8)

            if len(image.shape) < 4:
                image = image.unsqueeze(0)
            tensor_batch.append(image)
            label_batch.append(torch.Tensor([label]).to(torch.int64))

            tensor_size += math.prod(image.shape)

            if tensor_size - last_print_size > 1024 * 1024 * 100:
                last_print_size = tensor_size

                print(f"size: {tensor_size:,}")

            if tensor_size >= max_megabyte * 1024 * 1024:
                break

    except KeyboardInterrupt:
        if not get_answer("store dataset"):
            return

    tensor_batch = torch.cat(tensor_batch)
    print(f"saving {tensor_batch.shape} to {output_filename}.pt")
    torch.save(tensor_batch, f"{output_filename}.pt")

    label_batch = torch.cat(label_batch)
    print(f"saving {label_batch.shape} to {output_filename}-labels.pt")
    torch.save(label_batch, f"{output_filename}-labels.pt")


def store_ifs_dataset(
        num_classes: int,
        num_variations: int,
        seed: int,
        shape: Tuple[int, int] = (128, 128),
        process_index: int = 0,
):
    dataset_name = f"./datasets/ifs-1x{shape[-2]}x{shape[-1]}-uint8-{num_classes}x{num_variations}-seed{seed}"

    ds = IFSClassIterableDataset(
        num_classes=num_classes,
        num_instances_per_class=num_variations,
        start_seed=seed,
        variation_seed=seed,
        shape=shape,
        num_iterations=10_000, alpha=.15,
        #num_iterations=500, alpha=1,
        parameter_variation=0.05,
        parameter_variation_max=0.09,
        alpha_variation=0.12,
        patch_size_variations=[1, 1, 1, 3, 3, 5],
        num_iterations_variation=10_000,
        image_filter=ImageFilter(
            min_mean=0.2,
            max_mean=0.27,
            #min_blurred_compression_ratio=.6,
        ),
        filter_shape=(32, 32),
        filter_num_iterations=1000,
        filter_alpha=1.
    )

    store_dataset(ds, dataset_name, tqdm_label=f"seed{seed}", tqdm_position=process_index)


def _store_ifs_dataset(kwargs):
    store_ifs_dataset(**kwargs)


def combine_datasets(
        num_classes: int,
        num_variations: int,
        shape: Tuple[int, int] = (128, 128),
        **kwargs
):
    dataset_name = f"ifs-1x{shape[-2]}x{shape[-1]}-uint8-{num_classes}x{num_variations}"
    dataset_names = []
    for filename in sorted(Path("./datasets/").glob("*.pt")):
        if filename.name.startswith(dataset_name):
            dataset_names.append(filename.name)

    image_names = {}
    label_names = {}
    for name in dataset_names:
        seed_part = name[len(dataset_name):]
        match = re.match(r"-seed(\d+)", seed_part)
        if match:
            seed = match.groups()[0]
            if "-labels" in seed_part:
                label_names[seed] = name
            else:
                image_names[seed] = name

    database_names = [
        {"image": image_names[seed], "label": label_names[seed]}
        for seed in image_names
        if seed in label_names
    ]
    for entry in database_names:
        print(entry["image"], "&", entry["label"])

    if not get_answer("combine these"):
        return

    images = []
    labels = []
    for entry in database_names:
        images.append(torch.load(f"./datasets/{entry['image']}"))
        labels.append(torch.load(f"./datasets/{entry['label']}"))
    images = torch.cat(images)
    labels = torch.cat(labels)
    num_classes = len(set(labels.tolist()))

    print(f"images: {images.shape}, labels: {labels.shape}, num_classes: {num_classes}")

    dataset_name = f"ifs-1x{shape[-2]}x{shape[-1]}-uint8-{num_classes}x{num_variations}"

    for data, filename in (
            (images, f"./datasets/{dataset_name}.pt"),
            (labels, f"./datasets/{dataset_name}-labels.pt"),
    ):
        if Path(filename).exists():
            if not get_answer(f"overwrite {filename}"):
                continue

        print(f"saving {filename}")
        torch.save(data, filename)


def test_dataset(
        num_classes: int,
        num_variations: int,
        shape: Tuple[int, int] = (128, 128),
        **kwargs
):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    dataset_name = f"./datasets/ifs-1x{shape[-2]}x{shape[-1]}-uint8-{num_classes}x{num_variations}"
    print(f"testing {dataset_name}")
    images = torch.load(f"{dataset_name}.pt").numpy()
    images = images.reshape(images.shape[0], -1)
    labels = torch.load(f"{dataset_name}-labels.pt").numpy()

    print("pca...")
    pca = PCA(128)
    features = pca.fit_transform(images)

    print("classifier..")
    classifier = KNeighborsClassifier()
    classifier.fit(features, labels)
    predicted_labels = classifier.predict(features)

    num_correct = (labels == predicted_labels).astype(np.int8).sum()
    accuracy = num_correct / len(labels) * 100.
    print(f"correct/all: {num_correct}/{len(labels)} ({accuracy:.3f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, nargs="?", default="make",
        choices=["make", "combine", "test"],
    )
    parser.add_argument(
        "-s", "--seed", type=int, nargs="+", default=[0],
        help="The random seed for the image creation. Supply more than "
             "one seed to render datasets in parallel."
    )
    parser.add_argument(
        "-nc", "--num-classes", type=int, nargs="?", default=1000,
    )
    parser.add_argument(
        "-nv", "--num-variations", type=int, nargs="?", default=16,
    )
    kwargs = vars(parser.parse_args())
    print(json.dumps(kwargs, indent=2))

    mode = kwargs.pop("mode")
    if mode == "combine":
        combine_datasets(**kwargs)
        return
    elif mode == "test":
        test_dataset(**kwargs)
        return

    if len(kwargs["seed"]) == 1:
        kwargs["seed"] = kwargs["seed"][0]
        store_ifs_dataset(**kwargs)

    else:
        pool = Pool(len(kwargs["seed"]))
        arguments = []
        for i, seed in enumerate(kwargs["seed"]):
            arguments.append({
                **kwargs,
                "seed": seed,
                "process_index": i
            })

        pool.map(_store_ifs_dataset, arguments)


if __name__ == "__main__":
    main()
