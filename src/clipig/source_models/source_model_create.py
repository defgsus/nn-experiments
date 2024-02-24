from copy import deepcopy
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union, Generator, Dict, Type, Any

import yaml
import torch

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
