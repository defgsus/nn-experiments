import json
import os
import re
import math
import random
import itertools
import secrets
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

import numpy as np
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

# make all the local stuff available to experiments
from src import console
from src.util import *
from src.util.image import *
from src.util.module import *
from src.functional import *
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
from src.models.img2img import *
from src.models.util import *
from src.datasets import *
from src.datasets.aug import *
from src.datasets.generative import *
from src.algo import *
try:
    from experiments.datasets import *
except ImportError as e:
    warnings.warn(str(e))
    pass


RESERVED_MATRIX_KEYS = (
    "matrix_slug",
    "matrix_id",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_experiment_from_command_args():
    parser = argparse.ArgumentParser()
    Trainer.add_parser_args(parser)
    parser.add_argument(
        "command", type=str,
        choices=["run", "show", "results"],
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Load model from another checkpoint"
    )
    parser.add_argument(
        "-s", "--skip", type=bool, nargs="?", default=False, const=True,
        help="Skip experiment if checkpoint exists"
    )
    parser.add_argument(
        "-ic", "--include-column", type=str, nargs="*", default=[],
        help="one or more scalar values of the experiment results to include in the table"
    )
    parser.add_argument(
        "-ec", "--exclude-column", type=str, nargs="*", default=[],
        help="one or more columns to exclude from experiment results"
    )
    parser.add_argument(
        "-sc", "--sort-column", type=str, nargs="*", default=[],
        help="one or more columns to sort"
    )
    parser.add_argument(
        "-ac", "--average-column", type=str, default=None,
        help="Ignore value of this column and report average. E.g., add a variable "
             "`trial: [1, 2, 3, 4, 5]` to the experiment matrix and call `-ac trial` "
             "to display the average over all trials."
    )
    parser.add_argument(
        "-acs", "--average-column-std", type=str, nargs="+", default=[],
        help="Also display standard deviation for these columns"
    )

    args = vars(parser.parse_args())
    experiment_file = args.pop("experiment_name")

    run_experiment(experiment_file, extra_args=args)


def load_experiment_trainer(experiment_file, device: str = "auto"):
    return run_experiment(experiment_file, extra_args={
        "command": "get_trainer",
        "device": device,
    })


def run_experiment(filename: Union[str, Path], extra_args: dict):
    command = extra_args.pop("command")
    skip_existing = extra_args.pop("skip", False)
    exclude_columns = extra_args.pop("exclude_column", [])
    include_columns = extra_args.pop("include_column", [])
    sort_columns = extra_args.pop("sort_column", [])
    average_column = extra_args.pop("average_column", None)
    average_column_std = extra_args.pop("average_column_std", [])
    load_from_checkpoint = extra_args.pop("load", None)

    data = _load_yaml(filename)

    matrix_entries = get_matrix_entries(data.pop("matrix", None))
    if len(matrix_entries) > 1:
        print(f"\n{'running' if command == 'run' else 'showing'} {len(matrix_entries)} matrix experiments\n")

    experiment_results = []

    for matrix_entry in matrix_entries:

        data = _load_yaml(filename, matrix_entry)
        data.pop("matrix", None)
        _load = data.pop("load", None)
        if _load and not load_from_checkpoint:
            load_from_checkpoint = _load

        if extra_args:
            data.update(extra_args)

        if len(matrix_entries) > 1 and command != "results":
            print(f"\n--- matrix experiment '{data['experiment_name']}' ---\n")
            max_len = max(list(len(key) for key in matrix_entry.keys()))
            for key, value in matrix_entry.items():
                if key not in RESERVED_MATRIX_KEYS:
                    print(f"{key:{max_len}}: {value}")
            print()

        checkpoint_path = PROJECT_ROOT / "checkpoints" / data["experiment_name"]
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
        if command != "get_trainer":
            print(model)

        for key in ("encoder", "decoder"):
            if hasattr(model, key):
                print(f"{key} params: {num_module_parameters(getattr(model, key)):,}")

        if command not in ("run", "get_trainer"):
            continue

        trainer = trainer_klass(**kwargs)

        if not kwargs.get("reset"):
            if not trainer.load_checkpoint("best"):
                trainer.load_checkpoint()

        if command == "get_trainer":
            return trainer

        if load_from_checkpoint is not None:
            found_it = False
            for cp_filename in ("best.pt", "snapshot.pt"):
                cp_filename = Path(load_from_checkpoint) / cp_filename
                print(cp_filename, cp_filename.exists())
                if cp_filename.exists():
                    found_it = True
                    print(f"loading model checkpoint {cp_filename}")
                    checkpoint_data = torch.load(cp_filename)
                    model.load_state_dict(checkpoint_data["state_dict"])
                    break

            if not found_it:
                print(f"Did not find checkpoint in `{load_from_checkpoint}`")
                exit(-1)

        trainer.save_description()
        trainer.train()

        if snapshot_json_file.exists():
            experiment_results.append((matrix_entry, json.loads(snapshot_json_file.read_text())))

    if experiment_results:
        dump_experiments_results(
            experiment_results,
            include_columns=include_columns,
            exclude_columns=exclude_columns, sort_columns=sort_columns,
            average_column=average_column, average_column_std=average_column_std,
        )


def dump_experiments_results(
        experiment_results: List[Tuple[dict, dict]],
        include_columns: List[str],
        exclude_columns: List[str],
        sort_columns: List[str],
        average_column: Optional[str],
        average_column_std: List[str],
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

        for key in include_columns:
            row[key] = snapshot_data["scalars"][key]["value"]

        if snapshot_data.get("extra"):
            for key, value in snapshot_data["extra"].items():
                row[key] = value

        if snapshot_data.get("trainable_parameters"):
            row["model params"] = "{:,}".format(snapshot_data["trainable_parameters"][0])

        if snapshot_data.get("training_time"):
            seconds = snapshot_data["training_time"]
            row["train time (minutes)"] = seconds / 60.
            row["throughput"] = num_inputs / snapshot_data["training_time"]

        for key, value in row.items():
            if isinstance(value, (tuple, list)):
                row[key] = ",".join(str(v) for v in value)

        rows.append(row)

    #if max_loss is not None and max_loss != min_loss:
    #    for row in rows:
    #        for key, value in list(row.items()):
    #            if key.startswith("validation loss") and math.isfinite(value):
    #                length = int((value - min_loss) / (max_loss - min_loss) * 20 + 1)
    #                row["meter"] = "*" * length

    df = pd.DataFrame(rows)

    if average_column:
        df = group_df_column(df, average_column, average_column_std)

    for key in reversed(sort_columns):
        if key not in df.columns:
            for column in df.columns:
                if key in column:
                    key = column
                    break

        ascending = True
        if key.startswith("-"):
            key = key[1:]
            ascending = False
        elif key.endswith("-"):
            key = key[:-1]
            ascending = False
        try:
            df.sort_values(key, ascending=ascending, inplace=True, kind="stable")
        except KeyError as e:
            print("COLUMNS:", df.columns)
            raise

    df.loc[:, "train time (minutes)"] = df.loc[:, "train time (minutes)"].round(2)
    df.loc[:, "throughput"] = df.loc[:, "throughput"].map(lambda t: f"{int(t):,}/s")

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

    right_columns = ["model params", "throughput"]
    print(df.to_markdown(
        index=show_index,
        colalign=[
            "right" if c in right_columns else ("left" if df.iloc[:, i].dtype == np.object_ else "right")
            for i, c in enumerate(df.columns)
        ]
    ))


def group_df_column(
        df: pd.DataFrame,
        column: str,
        std_columns: List[str],
) -> pd.DataFrame:
    column_values = df.loc[:, column].unique()
    column_values = sorted(str(c) for c in column_values)
    column_values = sorted(column_values, key=lambda c: len(c), reverse=True)

    def _remove_column_ref(x):
        for c in column_values:
            x = x.replace(f"{column}:{c}", "")
        return x

    df["_id_without"] = df["matrix_slug"].map(_remove_column_ref)
    group = df.groupby("_id_without")
    df2 = group.mean(numeric_only=True)  # get mean of float values
    df2_std = group.std(numeric_only=True)
    df3 = group.max()                    # get all values (including strings)
    group_count = group.count()

    for c in df3.columns:
        if c not in df2.columns:
            df2.loc[:, c] = df3.loc[:, c]  # copy strings back

    dic = {
        f"num {column}s": group_count.loc[:, column]
    }
    for c in df3:
        # restore old column order
        dic[c] = df2.loc[:, c]

        # add min/max/std columns
        if c in std_columns:
            for name, extra_df in (
                    ("min", group.min()),
                    ("max", group.max()),
                    ("std", df2_std),
            ):
                if c in extra_df:
                    key = f"({name})"
                    if key in dic:
                        key = f"{key}REMOVE{c}"
                    dic[key] = extra_df.loc[:, c]

    df = pd.DataFrame(dic).reset_index().drop(["_id_without", column], axis=1)
    df.columns = df.columns.map(lambda c: c.split("REMOVE")[0])
    return df


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
                if construct_from_code(apply_parameter_matrix(
                    str(matrix["$filter"]).replace("\n", " "),
                    entry
                ))
            ]

    return matrix_entries


def get_matrix_slug(entry: dict) -> str:
    def _value_str(value):
        return str(value)[:96]

    slug = "_".join(
        f"{key}:{_value_str(value)}"
        for key, value in entry.items()
    )
    return "".join(
        c for c in slug
        if c.isalnum() or c in ".,-_:"
    )


_yaml_data_cache = {}
_yaml_text_cache = {}


def _load_yaml(filename: Union[str, Path], matrix_entry: Optional[Dict] = None):
    filename = str(filename)

    if not matrix_entry:
        if filename not in _yaml_data_cache:
            with open(filename) as fp:
                _yaml_data_cache[filename] = yaml.load(fp, YamlLoader)
        data = _yaml_data_cache[filename]

    else:
        if filename not in _yaml_text_cache:
            _yaml_text_cache[filename] = Path(filename).read_text()
        text = _yaml_text_cache[filename]

        text = apply_parameter_matrix(text, {
            **matrix_entry,
            "PATH": str(PROJECT_ROOT),
        })

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
        "extra_description_values": {
            "extra": {},
        },
    }

    # add yaml config
    locals_ = {
        "EXTRA_VALUES": {}
    }
    for key, value in data.items():
        if key in (
                "model",
                "train_set",
                "validation_set",
        ):
            value = construct_from_code(value, locals_)

        # interpret multiline strings as code
        elif isinstance(value, str) and "\n" in value:
            value = construct_from_code(value, locals_)

        kwargs[key] = value

    kwargs["extra_description_values"]["extra"].update(locals_["EXTRA_VALUES"])

    trainer_class = kwargs.pop("trainer", None)
    train_set = kwargs.pop("train_set")
    validation_set = kwargs.pop("validation_set", None)
    batch_size = kwargs.pop("batch_size")
    learnrate = kwargs.pop("learnrate")
    optimizer = kwargs.pop("optimizer")
    scheduler = kwargs.pop("scheduler", None)
    num_workers = kwargs.pop("num_workers", 1)
    dataloader_collate_fn = kwargs.pop("dataloader_collate_fn", None)

    if trainer_class is not None:
        trainer_class = get_class(trainer_class)
    else:
        trainer_class = Trainer

    kwargs["data_loader"] = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not isinstance(train_set, IterableDataset),
        num_workers=num_workers,
        collate_fn=dataloader_collate_fn,
    )

    validation_batch_size = kwargs.pop("validation_batch_size", None)
    if validation_set is not None:
        kwargs["validation_loader"] = DataLoader(
            validation_set,
            batch_size=validation_batch_size or batch_size,
            collate_fn=dataloader_collate_fn,
        )

    kwargs["optimizers"] = [
        construct_optimizer(kwargs["model"], learnrate, optimizer)
    ]
    if scheduler is not None:
        if not kwargs.get("max_inputs"):
            raise ValueError(f"`max_inputs` must be defined when using `scheduler`")
        kwargs["schedulers"] = [
            construct_scheduler(kwargs["optimizers"][0], scheduler, {**kwargs, "batch_size": batch_size})
        ]

    if not kwargs["extra_description_values"]["extra"]:
        del kwargs["extra_description_values"]

    return trainer_class, kwargs


def construct_optimizer(model: nn.Module, learnrate: float, parameter: str):
    klass = getattr(torch.optim, parameter)
    return klass(model.parameters(), lr=float(learnrate))


def construct_scheduler(optimizer: torch.optim.Optimizer, parameter: str, kwargs: dict):
    import src.scheduler
    klass = getattr(src.scheduler, parameter, None)
    if klass is None:
        klass = getattr(torch.optim.lr_scheduler, parameter)
    # print("SCHEDULER", klass, kwargs["max_inputs"] // kwargs["batch_size"])
    return klass(optimizer, kwargs["max_inputs"] // kwargs["batch_size"])


def construct_from_code(code: Any, locals_: Optional[dict] = None) -> Any:
    """
    Returns last statement

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
        _locals = {} if locals_ is None else locals_
        _globals = globals().copy()
        exec(compile(block, '<string>', mode='exec'), _globals, _locals)
        _globals.update(_locals)
        result = eval(compile(last, '<string>', mode='eval'), _globals, _locals)
        return result

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

        filename = str(self._root / self.construct_scalar(node))

        if filename not in _yaml_data_cache:
            with open(filename) as fp:
                _yaml_data_cache[filename] = yaml.load(fp, YamlLoader)
        return _yaml_data_cache[filename]
