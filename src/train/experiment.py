import json
import os
import re
import math
import random
import itertools
import argparse
import shutil
import sys
import warnings
import ast
from pathlib import Path
import importlib
from typing import List, Iterable, Tuple, Optional, Callable, Union, Generator, Dict, Type

import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util import *
from src.util.image import *
from src.train.trainer import *
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
from src.datasets import *
from src.algo import *


def run_experiment_from_command_args():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)

    args = vars(parser.parse_args())
    experiment_file = args.pop("experiment_name")

    with open(experiment_file) as fp:
        data = yaml.safe_load(fp)

    run_experiment({**data, **args})


def run_experiment(data: Union[dict, str, Path]):
    if isinstance(data, (str, Path)):
        with open(data) as fp:
            data = yaml.safe_load(fp)

    trainer_klass, kwargs = get_trainer_kwargs_from_dict(data)

    model = kwargs["model"]
    print(model)
    for key in ("encoder", "decoder"):
        if hasattr(model, key):
            print(f"{key} params: {num_module_parameters(getattr(model, key)):,}")

    trainer = trainer_klass(**kwargs)

    if not kwargs["reset"]:
        trainer.load_checkpoint()

    trainer.save_description()
    trainer.train()


def get_trainer_kwargs_from_dict(data: dict) -> Tuple[Type[Trainer], dict]:
    required_keys = (
        "experiment_name",
        "model",
        "train_set",

        "batch_size",
        "learnrate",
        "optimizer",
#        "scheduler",
    )
    for key in required_keys:
        if not data.get(key):
            raise ValueError(f"Required parameter `{key}` is missing")

    globals_ = data.pop("globals", None)
    if globals_:
        for key, value in globals_.items():
            print("X", repr(value), repr(construct_from_code(str(value))))
            globals()[key] = construct_from_code(str(value))

    kwargs = {
        "num_epochs_between_validations": 1,
    }
    for key, value in data.items():

        if key in (
                "model",
                "train_set",
                "validation_set",
        ):
            value = construct_from_code(value)

        kwargs[key] = value

    trainer_class = kwargs.pop("trainer", None)
    train_set = kwargs.pop("train_set")
    validation_set = kwargs.pop("validation_set", None)
    batch_size = kwargs.pop("batch_size")
    learnrate = kwargs.pop("learnrate")
    optimizer = kwargs.pop("optimizer")
    scheduler = kwargs.pop("scheduler", None)

    if trainer_class is not None:
        trainer_class = get_class(trainer_class)
    else:
        trainer_class = Trainer

    kwargs["data_loader"] = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not isinstance(train_set, IterableDataset),
    )

    if validation_set is not None:
        kwargs["validation_loader"] = DataLoader(validation_set, batch_size=batch_size)

    kwargs["optimizers"] = [
        construct_optimizer(kwargs["model"], learnrate, optimizer)
    ]
    if scheduler is not None:
        if not kwargs.get("max_inputs"):
            raise ValueError(f"`max_inputs` must be defined when using `scheduler`")
        kwargs["schedulers"] = [
            construct_scheduler(kwargs["optimizers"][0], scheduler, {**kwargs, "batch_size": batch_size})
        ]
    return trainer_class, kwargs


def construct_optimizer(model: nn.Module, learnrate: float, parameter: str):
    klass = getattr(torch.optim, parameter)
    return klass(model.parameters(), lr=learnrate)


def construct_scheduler(optimizer: torch.optim.Optimizer, parameter: str, kwargs: dict):
    klass = getattr(torch.optim.lr_scheduler, parameter)
    return klass(optimizer, kwargs["max_inputs"] // kwargs["batch_size"])


def construct_from_code(code: str):
    """
    https://stackoverflow.com/questions/39379331/python-exec-a-code-block-and-eval-the-last-line/39381428#39381428
    """
    block = ast.parse(code, mode='exec')

    # assumes last node is an expression
    last = ast.Expression(block.body.pop().value)

    try:
        _locals = {}
        exec(compile(block, '<string>', mode='exec'), globals(), _locals)
        return eval(compile(last, '<string>', mode='eval'), globals(), _locals)

    except:
        print("\n".join(
            f"{i + 1:3}: {line}"
            for i, line in enumerate(code.splitlines())
        ))
        raise


def get_class(arg: str):
    args = arg.split(".")
    if len(args) > 1:
        path = ".".join(args[:-1])
        module = importlib.import_module(path)
        return getattr(module, args[-1])

    return globals()[arg]
