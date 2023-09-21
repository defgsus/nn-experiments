from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset

from src.datasets.transform import TransformIterableDataset
from src.datasets.image_folder import ImageFolderIterableDataset
from src.datasets.image_patch import ImagePatchIterableDataset
from src.datasets.shuffle import IterableShuffle
from src.util.image import set_image_channels
