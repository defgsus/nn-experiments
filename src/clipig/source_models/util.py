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


PROJECT_PATH = Path(__file__).resolve().parent.parent.parent.parent
AUTOENCODER_PATH = Path(__file__).resolve().parent.parent / "models/autoencoder"
PROCESS_PATH = Path(__file__).resolve().parent.parent / "models/process"


def construct_from_code(code: Any):
    """
    https://stackoverflow.com/questions/39379331/python-exec-a-code-block-and-eval-the-last-line/39381428#39381428
    """
    if not isinstance(code, str):
        return code

    block = ast.parse(code, mode='exec')

    # assumes last node is an expression
    stmt = block.body.pop()
    try:
        last = ast.Expression(stmt.value)
    except AttributeError as e:
        raise AttributeError(f"{e}, in code:\n{code}") from e

    try:
        _locals = {}
        _globals = globals().copy()
        exec(compile(block, '<string>', mode='exec'), _globals, _locals)
        _globals.update(_locals)
        return eval(compile(last, '<string>', mode='eval'), _globals, _locals)

    except:
        print("\n".join(
            f"{i + 1:3}: {line}"
            for i, line in enumerate(code.splitlines())
        ))
        raise


_model_stack: List[Tuple[Path, Tuple[nn.Module, dict]]] = []


def load_model_from_yaml(filename: Union[str, Path]) -> Tuple[nn.Module, dict]:
    filename = Path(filename)

    for stack in _model_stack:
        if stack[0] == filename:
            return stack[1]

    with filename.open() as fp:
        data = yaml.safe_load(fp)

    model = construct_from_code(data["model"])
    checkpoint_filename = data.get("checkpoint")
    if checkpoint_filename:
        checkpoint_filename = get_full_yaml_filename(checkpoint_filename, filename.parent)

        state_dict = torch.load(checkpoint_filename)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)

    if len(_model_stack) > 5:
        _model_stack.pop(0)

    _model_stack.append((filename, (model, data)))

    return model, data


def get_full_yaml_filename(
    filename: Union[str, Path],
    yaml_directory: Union[str, Path],
) -> Path:
    filename = Path(
        str(filename).replace("$PROJECT", str(PROJECT_PATH).rstrip(os.path.sep))
    )
    if not filename.is_absolute():
        filename = Path(yaml_directory) / filename

    return filename

