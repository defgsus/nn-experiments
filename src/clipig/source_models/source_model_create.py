import json
import os
import re
import math
import random
import itertools
import shutil
import argparse
import sys
import warnings
import ast
from copy import deepcopy
from io import StringIO
from pathlib import Path
import importlib
from typing import List, Iterable, Tuple, Optional, Callable, Union, Generator, Dict, Type, Any

import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules
import torch.nn.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import torchvision.models
from torchvision.utils import make_grid
import pandas as pd

# make all this stuff available to yaml code snippets
from src import console
from src.util import *
from src.util.image import *
from src.train import *
from src.models.util import *
from src.models.ca import *
from src.models.cnn import *
from src.models.decoder import *
from src.models.encoder import *
from src.models.generative import *
from src.models.rbm import *
from src.models.recursive import *
from src.models.transform import *
from src.models.vae import *
from src.models.clip import ClipSingleton
from src.models.loss import *
from src.models.util import *
from src.algo import *

from . import base
from .util import construct_from_code, PROJECT_PATH, AUTOENCODER_PATH, get_full_yaml_filename


def create_source_model(config: dict, device: torch.device):
    if config["name"] not in base.source_models:
        raise ValueError(f"Unknown source_model '{config['name']}'")

    klass = base.source_models[config["name"]]

    kwargs = deepcopy(config["params"])

    if klass.IS_AUTOENCODER:
        _create_autoencoder(kwargs)

    model = klass(**kwargs).to(device)
    return model


def _create_autoencoder(kwargs: dict):
    ae_config = _get_autoencoder_config(kwargs["autoencoder"])

    kwargs["autoencoder_shape"] = ae_config["shape"]
    kwargs["code_size"] = ae_config["code_size"]

    kwargs["autoencoder"] = model = construct_from_code(ae_config["model"])

    if ae_config.get("checkpoint"):
        filename = get_full_yaml_filename(ae_config["checkpoint"], AUTOENCODER_PATH)

        state_dict = torch.load(filename)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)

    model.eval()


def _get_autoencoder_config(name: str):
    filename = AUTOENCODER_PATH / f"{name}.yaml"
    with filename.open() as fp:
        return yaml.safe_load(fp)
