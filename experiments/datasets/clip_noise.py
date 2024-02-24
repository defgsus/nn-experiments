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


class ClipNoiseDataset(BaseIterableDataset):
    def __init__(
            self,
            patch_size: Tuple[int, int],
            interleave_images: int = 1,
    ):
        self._patch_size = patch_size
        self._interleave_images = interleave_images
        self._directory_orig = Path(__file__).resolve().parent.parent.parent / "datasets/declipig/original"
        self._directory_noisy = Path(__file__).resolve().parent.parent.parent / "datasets/declipig/clipped07"
        #self._directory_noisy = Path("../datasets/diffusion-clip-noised/")

    def __iter__(self):
        ps = self._patch_size
        image_pairs = []
        image_pair_iterable = self._iter_image_pairs()
        iter_count = -1
        while True:
            iter_count += 1
            while len(image_pairs) < self._interleave_images:
                try:
                    image_orig, image_noisy = next(image_pair_iterable)
                except StopIteration:
                    break

                size = image_orig.shape[-2:]
                count = (size[-2] // ps[-2]) * (size[-1] // ps[-1])
                count *= 10

                image_pairs.append({"count": count, "images": (image_orig, image_noisy)})

            if not image_pairs:
                break

            pair_index = iter_count % len(image_pairs)
            image_orig, image_noisy = image_pairs[pair_index]["images"]

            size = image_orig.shape[-2:]

            pos = (
                random.randrange(0, size[-2] - ps[-2]),
                random.randrange(0, size[-2] - ps[-2])
            )

            patch_orig = image_orig[:, pos[-2]: pos[-2] + ps[-2], pos[-1]: pos[-1] + ps[-1]]
            patch_noisy = image_noisy[:, pos[-2]: pos[-2] + ps[-2], pos[-1]: pos[-1] + ps[-1]]

            yield patch_orig, patch_noisy

            image_pairs[pair_index]["count"] -= 1
            if image_pairs[pair_index]["count"] <= 0:
                image_pairs.pop(pair_index)

    def _iter_image_pairs(self):
        for filename in sorted(self._directory_noisy.glob("*.jpeg")):
            image_noisy = VF.to_tensor(PIL.Image.open(filename))
            image_orig = VF.to_tensor(PIL.Image.open(
                self._directory_orig / filename.name #[:-4]
            ))
            yield image_orig, image_noisy
