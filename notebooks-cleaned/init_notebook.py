import sys
sys.path.append("..")

import io
import os
import random
import math
import time
import json
import shutil
import yaml
from io import BytesIO
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Dict

import pandas as pd
import numpy as np

import PIL.Image
import PIL.ImageDraw
import plotly
import plotly.express as px
plotly.io.templates.default = "plotly_dark"
pd.options.plotting.backend = "plotly"

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid
from IPython.display import display

from src.util import *
from src.util.image import *
from src.util.params import *
from src.datasets import *
from src.algo import *
from src.models.decoder import *
from src.models.transform import *
from src.models.util import *
from experiments import datasets
from experiments.denoise.resconv import ResConv

def resize(img, scale: float, mode: VF.InterpolationMode = VF.InterpolationMode.NEAREST):
    if isinstance(img, PIL.Image.Image):
        shape = (img.height, img.width)
    else:
        shape = img.shape[-2:]
    return VF.resize(img, [max(1, int(s * scale)) for s in shape], mode, antialias=False)
