import math
import json
import random
import fnmatch
from pathlib import Path
from functools import partial
from typing import List, Iterable, Tuple, Optional, Callable, Union

import pandas as pd
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


class PixelartDataset(BaseDataset):

    LABELS = [
        'creature',
        'wall',
        'tree',
        'carpet',
        'grass',
        'rock',
        'water',
        'wood',
        'sand',
        'roof',
        'sword',
        'cobblestone',
        'plant',
        'platform',
        'stairs',
        'shelf',
        'block',
        'axe',
        'food',
        'door',
        'dirt',
        'window',
        'pipe',
        'floor',
        'table',
        'bridge',
        'stone',
        'bed',
        'fire',
        'other',
    ]

    def __init__(
            self,
            shape: Tuple[int, int, int] = (3, 32, 32),
            with_clip_embedding: bool = False,
            normalized_clip_embedding: bool = True,
    ):
        self._out_shape = shape
        self._with_clip_embedding = with_clip_embedding
        self._normalized_clip_embedding = normalized_clip_embedding
        self._patch_dataset = None
        self._label_to_id = {l: i for i, l in enumerate(self.LABELS)}
        self._fallback_id = self._label_to_id["other"]
        self._embeddings = None

    def __len__(self):
        self._lazy_load()
        return self._meta["count"]

    def _lazy_load(self):
        if self._patch_dataset is None:
            path = Path("~/prog/python/github/pixelart-dataset/datasets/v2/").expanduser()
            self._meta = json.loads((path / "tiles.json").read_text())
            patch_shape = (self._out_shape[0], *self._meta["shape"])
            self._patch_dataset = ImagePatchDataset(patch_shape, path / "tiles.png")
            self._patch_df = pd.read_csv(path / "tiles.csv")
            if self._with_clip_embedding:
                self._embeddings = torch.load(path / "clip-vit-b32-embeddings.pt")
                if self._normalized_clip_embedding:
                    self._embeddings /= torch.norm(self._embeddings, dim=1, keepdim=True)

    def __getitem__(self, index: int):
        self._lazy_load()
        item = self._patch_dataset[index]
        label = self._patch_df.iloc[index]["label"]

        if label not in self._label_to_id:
            for base_label in self.LABELS:
                if base_label in label:
                    self._label_to_id[label] = self._label_to_id[base_label]
                    break

        label_id = self._label_to_id.get(label, self._fallback_id)

        if not self._with_clip_embedding:
            return item, label_id
        else:
            return item, label_id, self._embeddings[index]
