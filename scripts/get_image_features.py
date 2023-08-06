import math
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader

from src.util import ImageCompressionRatio


@torch.no_grad()
def main(
        dataset_name="./datasets/kali-uint8-64x64.pt",
        max_count: int = 0,
):
    dataset_name = Path(dataset_name)

    ratioer = ImageCompressionRatio()
    def get_features(image: torch.Tensor) -> dict:
        image_blurred = VF.gaussian_blur(image, [21, 21], [20., 20.])
        return {
            **ratioer.all(image, prefix="cr-"),
            **ratioer.all(image_blurred, prefix="cr-", suffix="-blur"),
        }

    dataset = TensorDataset(torch.load(dataset_name))
    dataloader = DataLoader(dataset)

    feature_rows = []
    try:
        for image in tqdm(dataloader):
            if isinstance(image, (tuple, list)):
                image = image[0]

            if len(image.shape) == 4:
                image = image.squeeze(0)

            features = get_features(image)
            feature_rows.append(features)

            if max_count and len(feature_rows) >= max_count:
                break

    except KeyboardInterrupt:
        pass

    features_df = pd.DataFrame(feature_rows)

    filename_stem = dataset_name.name[:dataset_name.name.rindex('.')]
    output_filename = dataset_name.with_name(f"{filename_stem}-features.df")
    print(f"saving {features_df.shape} to {output_filename}")
    features_df.to_pickle(output_filename)


if __name__ == "__main__":
    main()
