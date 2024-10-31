from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.datasets import *
from src.transforms import *


def boulderdash_dataset_32x32(
        validation: bool,
        signed: bool = True,
):
    path = Path(__file__).resolve().parent.parent.parent / "datasets"
    if validation:
        filename_part = "boulderdash-32x32-5000-validation"
    else:
        filename_part = "boulderdash-32x32-60000"

    ds = WrapDataset(TensorDataset(
        torch.load(path / f"{filename_part}-map1.pt"),
        torch.load(path / f"{filename_part}-map2.pt"),
    )).transform(dtype=torch.float, transform_all=True)
    if signed:
        ds = ds.transform([lambda x: x * 2 - 1.], transform_all=True)
    return ds
