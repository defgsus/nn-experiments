import math
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader
import clip

from src.util import ImageCompressionRatio
from src.datasets import *


@torch.no_grad()
def get_features(
        dataloader: DataLoader,
        model: nn.Module,
        preproc: Callable,
        max_count: int = 0,
        max_megabyte: int = 1024,
):
    feature_rows = []
    byte_count = 0
    try:
        with tqdm(desc="get features", total=len(dataloader.dataset)) as progress:
            for image in dataloader:
                if isinstance(image, (tuple, list)):
                    image = image[0]

                image = image.half().to("cuda")

                features = model(preproc(image))
                feature_rows.append(features.cpu().float())

                progress.update(features.shape[0])
                byte_count += math.prod(features.shape) * 4

                if max_count and progress.n >= max_count:
                    break

                if max_megabyte and byte_count >= max_megabyte * 1024 * 1024:
                    break

    except KeyboardInterrupt:
        pass

    return torch.cat(feature_rows)


def main():
    if 0:
        dataset = TransformDataset(
            TensorDataset(torch.load(f"./datasets/kali-uint8-64x64.pt")),
            dtype=torch.float, multiply=1. / 255.,
            #transforms=[lambda i: i.repeat(3, 1, 1)],
        )

    if 0:
        dataset = TransformDataset(
            TensorDataset(torch.load(f"./datasets/pattern-1x64x64-uint.pt")),
            dtype=torch.float, multiply=1. / 255.,
            transforms=[lambda i: i.repeat(3, 1, 1)],
        )

    if 1:
        dataset = TransformDataset(
            TensorDataset(torch.load(f"./datasets/photos-64x64-bcr03.pt")),
            #dtype=torch.float, multiply=1. / 255.,
            #transforms=[lambda i: i.repeat(3, 1, 1)],
        )

    one_image = dataset[0][0]
    assert one_image.shape == (3, 64, 64), one_image.shape
    assert one_image.dtype == torch.float, one_image.dtype

    model, preproc = clip.load("ViT-B/32")
    print(preproc)

    features = get_features(
        dataloader=DataLoader(dataset, batch_size=64),
        model=model.visual,
        preproc=VT.Compose([
            preproc.transforms[0],
            preproc.transforms[1],
            preproc.transforms[-1],
        ]),
        #max_count=100,
    )
    print(features.shape)

    torch.save(features, "./datasets/photos-64x64-bcr03-CLIP.pt")


if __name__ == "__main__":
    main()
