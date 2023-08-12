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

import PIL.Image
import PIL.ImageDraw
import plotly
import plotly.express as px
plotly.io.templates.default = "plotly_dark"
import numpy as np
import pandas as pd

from src.datasets import *
from src.util.image import *
from src.util import ImageFilter
from src.algo import IFS


def get_answer(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} (Y/n)? ")
        if answer in ("n", "N"):
            return False
        if answer in ("", "y", "Y"):
            return True


class IFSClassIterableDataset(IterableDataset):
    def __init__(
            self,
            shape: Tuple[int, int],
            num_classes: int,
            num_instances_per_class: int = 1,
            num_iterations: int = 10_000,
            seed: Optional[int] = None,
            alpha: float = 0.1,
            patch_size: int = 1,
            parameter_variation: float = 0.03,
            parameter_variation_max: Optional[float] = None,
            alpha_variation: float = 0.05,
            patch_size_variations: Optional[Iterable[int]] = None,
            num_iterations_variation: int = 0,
            image_filter: Optional[ImageFilter] = None,
            filter_num_iterations: Optional[int] = None,
            filter_shape: Optional[Tuple[int, int]] = None,
            filter_alpha: Optional[float] = None,
    ):
        self.shape = shape
        self.num_classes = num_classes
        self.num_instances_per_class = num_instances_per_class
        self.num_iterations = num_iterations
        self.seed = seed if seed is not None else random.randint(0, 1e10)
        self.alpha = alpha
        self.patch_size = patch_size
        self.parameter_variation = parameter_variation
        self.parameter_variation_max = parameter_variation_max
        self.alpha_variation = alpha_variation
        self.patch_size_variations = list(patch_size_variations) if patch_size_variations is not None else None
        self.num_iterations_variation = num_iterations_variation

        self.image_filter = image_filter
        self.filter_num_iterations = filter_num_iterations
        self.filter_shape = filter_shape
        self.filter_alpha = filter_alpha

        self.rng = np.random.Generator(np.random.MT19937(
            seed if seed is not None else random.randint(0, int(1e10))
        ))
        self.rng.bytes(42)

    def __len__(self) -> int:
        return self.num_classes * self.num_instances_per_class

    def _iter_class_seeds(self) -> Generator[int, None, None]:
        class_index = 0
        class_count = 0
        while class_count < self.num_classes:
            seed = class_index ^ self.seed

            ifs = IFS(seed=seed)
            class_index += 1

            if self.image_filter is not None:

                image = torch.Tensor(ifs.render_image(
                    shape=self.filter_shape or self.shape,
                    num_iterations=self.filter_num_iterations or self.num_iterations,
                    alpha=self.filter_alpha or self.alpha,
                    patch_size=self.patch_size,
                ))
                if not self.image_filter(image):
                    continue

            yield seed
            class_count += 1

    def __iter__(self) -> Generator[Tuple[torch.Tensor, int], None, None]:
        for class_index, seed in enumerate(self._iter_class_seeds()):

            instance_count = 0
            base_mean = None
            while instance_count < self.num_instances_per_class:
                ifs = IFS(seed=seed)

                alpha = self.alpha
                patch_size = self.patch_size
                num_iterations = self.num_iterations

                if instance_count > 0:
                    t = (instance_count + 1) / self.num_instances_per_class

                    amt = self.parameter_variation
                    if self.parameter_variation_max is not None:
                        amt = amt * (1. - t) + t * self.parameter_variation_max

                    ifs.parameters += amt* self.rng.uniform(-1., 1., ifs.parameters.shape)
                    alpha = max(.001, alpha + self.alpha_variation * self.rng.uniform(-1., 1.))
                    if self.patch_size_variations is not None:
                        patch_size = self.patch_size_variations[self.rng.integers(len(self.patch_size_variations))]
                    if self.num_iterations_variation:
                        num_iterations += self.rng.integers(self.num_iterations_variation)

                image = torch.Tensor(ifs.render_image(
                    shape=self.shape,
                    num_iterations=num_iterations,
                    alpha=alpha,
                    patch_size=patch_size,
                ))

                if base_mean is None:
                    base_mean = image.mean()
                else:
                    mean = image.mean()
                    if mean < base_mean / 1.5:
                        continue

                yield image, seed
                instance_count += 1


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
        num_classes=num_classes, num_instances_per_class=num_variations, seed=seed,
        shape=(128, 128), num_iterations=10_000, alpha=.15,
        #shape=(32, 32), num_iterations=1_000, alpha=1,
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

    print(f"images: {images.shape}, labels: {labels.shape}")

    dataset_name = f"ifs-1x{shape[-2]}x{shape[-1]}-uint8-{images.shape[0] // num_variations}x{num_variations}"

    for data, filename in (
            (images, f"./datasets/{dataset_name}.pt"),
            (labels, f"./datasets/{dataset_name}-labels.pt"),
    ):
        if Path(filename).exists():
            if not get_answer(f"overwrite {filename}"):
                continue

        print(f"saving {filename}")
        torch.save(data, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, nargs="?", default="make",
        choices=["make", "combine"],
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
