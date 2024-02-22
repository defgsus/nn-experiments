import math
import argparse
import random
from pathlib import Path
from functools import partial
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid

from src.datasets import *
from src.util.image import *
from .image_patch import _scales_from_image_shape


# currently about 3.8M patches
def kali_patch_dataset(
        shape: Tuple[int, int, int],
        path: Union[str, Path] = Path(__file__).resolve().parent.parent.parent / "db/images/kali",
        file_shuffle: bool = True,
):
    return make_image_patch_dataset(
        #verbose_image=True,
        path=path,

        recursive=True,
        shape=shape,
        max_image_bytes=1024 * 1024 * 1024 * 1,
        scales=partial(_scales_from_image_shape, shape, [2., 1., 1./2., 1./5, 1./10, 1./20., 1./30.]),
        stride=5,#_stride,
        interleave_images=20,
        #image_shuffle=5,
        patch_shuffle=10_000,
        file_shuffle=file_shuffle,
    )
