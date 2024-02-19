import os
import ast
from pathlib import Path
from typing import Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import yaml

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


def load_model_from_yaml(filename: Union[str, Path]) -> Tuple[nn.Module, dict]:
    filename = Path(filename)
    with filename.open() as fp:
        data = yaml.safe_load(fp)

    model = construct_from_code(data["model"])
    checkpoint_filename = data.get("checkpoint")
    if checkpoint_filename:
        checkpoint_filename = Path(
            checkpoint_filename
            .replace("$PROJECT", str(PROJECT_PATH).rstrip(os.path.sep))
        )
        if not checkpoint_filename.is_absolute():
            checkpoint_filename = filename.parent / checkpoint_filename

        state_dict = torch.load(checkpoint_filename)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)

    return model, data
