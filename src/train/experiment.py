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
from src.datasets import *
from src.algo import *
try:
    from experiments.datasets import *
except ImportError:
    pass


RESERVED_MATRIX_KEYS = (
    "matrix_slug",
    "matrix_id",
)


def run_experiment_from_command_args():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    parser.add_argument(
        "command", type=str,
        choices=["run", "show", "results"],
    )
    parser.add_argument(
        "-s", "--skip", type=bool, nargs="?", default=False, const=True,
        help="Skip experiment if checkpoint exists"
    )
    parser.add_argument(
        "-ec", "--exclude-column", type=str, nargs="*", default=[],
        help="one or more columns to exclude from experiment results"
    )
    parser.add_argument(
        "-sc", "--sort-column", type=str, nargs="*", default=[],
        help="one or more columns to sort"
    )

    args = vars(parser.parse_args())
    experiment_file = args.pop("experiment_name")

    run_experiment(experiment_file, extra_args=args)


def run_experiment(filename: Union[str, Path], extra_args: dict):
    command = extra_args.pop("command")
    skip_existing = extra_args.pop("skip")
    exclude_columns = extra_args.pop("exclude_column")
    sort_columns = extra_args.pop("sort_column")

    data = _load_yaml(filename)

    matrix_entries = get_matrix_entries(data.pop("matrix", None))
    if len(matrix_entries) > 1:
        print(f"\n{'running' if command == 'run' else 'showing'} {len(matrix_entries)} matrix experiments\n")

    experiment_results = []

    for matrix_entry in matrix_entries:

        data = _load_yaml(filename, matrix_entry)
        data.pop("matrix", None)
        if extra_args:
            data.update(extra_args)

        if len(matrix_entries) > 1:
            print(f"\n--- matrix experiment '{data['experiment_name']}' ---\n")
            max_len = max(list(len(key) for key in matrix_entry.keys()))
            for key, value in matrix_entry.items():
                if key not in RESERVED_MATRIX_KEYS:
                    print(f"{key:{max_len}}: {value}")
            print()

        checkpoint_path = Path("./checkpoints") / data["experiment_name"]
        snapshot_json_file = checkpoint_path / "snapshot.json"

        if command == "results":
            if snapshot_json_file.exists():
                experiment_results.append((matrix_entry, json.loads(snapshot_json_file.read_text())))
            continue

        trainer_klass, kwargs = get_trainer_kwargs_from_dict(data)

        if command == "show":
            continue

        if skip_existing:
            if (checkpoint_path / "snapshot.pt").exists():
                print("skipping", data["experiment_name"])
                continue

        model = kwargs["model"]
        print(model)
        for key in ("encoder", "decoder"):
            if hasattr(model, key):
                print(f"{key} params: {num_module_parameters(getattr(model, key)):,}")

        if command != "run":
            continue

        trainer = trainer_klass(**kwargs)

        if not kwargs["reset"]:
            trainer.load_checkpoint()

        trainer.save_description()
        trainer.train()

        if snapshot_json_file.exists():
            experiment_results.append((matrix_entry, json.loads(snapshot_json_file.read_text())))

    if experiment_results:
        dump_experiments_results(experiment_results, exclude_columns=exclude_columns, sort_columns=sort_columns)


def dump_experiments_results(
        experiment_results: List[Tuple[dict, dict]],
        exclude_columns: List[str],
        sort_columns: List[str],
):
    rows = []
    min_loss = None
    max_loss = None
    for matrix_entry, snapshot_data in experiment_results:
        row = deepcopy(matrix_entry)
        num_inputs = snapshot_data["num_inputs"]
        validation_loss = snapshot_data["validation_loss"]

        if snapshot_data.get("max_inputs"):
            num_inputs = min(num_inputs, snapshot_data["max_inputs"])

        row[f"validation loss ({num_inputs:,} steps)"] = validation_loss
        if max_loss is None:
            max_loss = min_loss = validation_loss
        else:
            max_loss = max(max_loss, validation_loss)
            min_loss = min(min_loss, validation_loss)
        rows.append(row)

    if max_loss is not None and max_loss != min_loss:
        for row in rows:
            for key, value in list(row.items()):
                if key.startswith("validation loss") and math.isfinite(value):
                    length = int((value - min_loss) / (max_loss - min_loss) * 20 + 1)
                    row["meter"] = "*" * length

    df = pd.DataFrame(rows)

    for key in reversed(sort_columns):
        ascending = True
        if key.startswith("-"):
            key = key[1:]
            ascending = False
        df.sort_values(key, ascending=ascending, inplace=True, kind="stable")

    for key in exclude_columns:
        try:
            df.drop(key, axis=1, inplace=True)
        except KeyError:
            pass

    try:
        df.set_index("matrix_id", inplace=True)
        show_index = True
    except KeyError:
        show_index = False

    print(df.to_markdown(index=show_index))


def get_matrix_entries(matrix: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not matrix:
        matrix_entries = [{}]
    else:
        for key in RESERVED_MATRIX_KEYS:
            if key in matrix:
                raise NameError(f"key '{key}' is not allowed in matrix")

        matrix_entries = list(iter_parameter_permutations(matrix, exclude_keys=["$filter"]))
        for i, entry in enumerate(matrix_entries):
            entry["matrix_slug"] = get_matrix_slug(entry)
            entry["matrix_id"] = f"{i + 1:0{int(math.ceil(math.log10(len(matrix_entries))))}}"

        if "$filter" in matrix:
            matrix_entries = [
                entry for entry in matrix_entries
                if construct_from_code(apply_parameter_matrix(str(matrix["$filter"]), entry))
            ]

    return matrix_entries


def get_matrix_slug(entry: dict) -> str:
    def _value_str(value):
        return str(value)[:40]

    slug = "_".join(
        f"{key}-{_value_str(value)}"
        for key, value in entry.items()
    )
    return "".join(
        c for c in slug
        if c.isalnum() or c in ".,-_"
    )


def _load_yaml(filename: str, matrix_entry: Optional[Dict] = None):
    if not matrix_entry:
        with open(filename) as fp:
            data = yaml.load(fp, YamlLoader)

    else:
        text = Path(filename).read_text()
        text = apply_parameter_matrix(text, matrix_entry)

        fp = StringIO(text)
        data = yaml.load(fp, YamlLoader)

    extends = data.pop("$extends", None)
    if extends:
        data = {
            **_load_yaml(Path(filename).parent / extends, matrix_entry=matrix_entry),
            **data,
        }

    return data


def apply_parameter_matrix(text: str, params: dict):
    re_variable = re.compile("\$\{([^}]+)}")
    def _repl(match):
        name = match.groups()[0]
        if name not in params:
            raise NameError(f"Variable ${{{name}}} is not defined in matrix")

        return str(params[name])

    return re_variable.sub(_repl, text)


def get_trainer_kwargs_from_dict(data: dict) -> Tuple[Type[Trainer], dict]:
    required_keys = (
        "experiment_name",
        "model",
        "train_set",

        "batch_size",
        "learnrate",
        "optimizer",
    )
    for key in required_keys:
        if not data.get(key):
            raise ValueError(f"Required parameter `{key}` is missing")

    globals_ = data.pop("globals", None)
    if globals_:
        for key, value in globals_.items():
            globals()[key] = construct_from_code(value)

    # defaults
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


def construct_from_code(code: Any):
    """
    https://stackoverflow.com/questions/39379331/python-exec-a-code-block-and-eval-the-last-line/39381428#39381428
    """
    if not isinstance(code, str):
        return code

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


class YamlLoader(yaml.SafeLoader):

    def __init__(self, stream):
        super().__init__(stream)
        self._root = Path(stream.name).parent if hasattr(stream, "name") else None

    def include(self, node):
        if self._root is None:
            raise ValueError(f"Can't !include in {type(self.stream).__name__}")

        filename = self._root / self.construct_scalar(node)

        with open(filename) as f:
            return yaml.load(f, YamlLoader)
