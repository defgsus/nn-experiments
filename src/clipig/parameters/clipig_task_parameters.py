import json
from pathlib import Path
from copy import deepcopy
from typing import Optional

import yaml


with (Path(__file__).resolve().parent / "clipig_task_parameters.yaml").open() as fp:
    _BASE_PARAMETERS = yaml.safe_load(fp)


def get_clipig_task_parameters():
    from .. import transformations
    from .. import source_models

    parameters = deepcopy(_BASE_PARAMETERS)

    parameters["transformations"] = {}
    for klass in transformations.transformations.values():
        parameters["transformations"][klass.NAME] = deepcopy(klass.PARAMS)

    parameters["source_models"] = {}
    for klass in source_models.source_models.values():
        parameters["source_models"][klass.NAME] = deepcopy(klass.PARAMS)

    for param in parameters["target"]:
        if param["name"] == "optimizer":
            param["choices"] = list(parameters["optimizers"].keys())

    # print(json.dumps(parameters, indent=2))
    return parameters


def get_complete_clipig_task_config(config: dict) -> dict:
    """
    Add all default values to the config dict.

    Returns new instance
    """
    config = deepcopy(config)

    parameters = get_clipig_task_parameters()

    for param in parameters["base"]:
        if param["name"] not in config:
            config[param["name"]] = param["default"]

    config["source_model"] = get_complete_clipig_source_model_config(
        config.get("source_model") or {},
        parameters=parameters,
    )

    if "targets" not in config:
        config["targets"] = []

    for target in config["targets"]:
        for param in parameters["target"]:
            if param["name"] not in target:
                target[param["name"]] = param["default"]

        if "transformations" not in target:
            target["transformations"] = []

        target["transformations"] = [
            get_complete_clipig_transformation_config(trans, parameters=parameters)
            for trans in target["transformations"]
        ]

    return config


def get_complete_clipig_transformation_config(trans: dict, parameters: Optional[dict] = None) -> dict:
    if parameters is None:
        parameters = get_clipig_task_parameters()

    trans = deepcopy(trans)

    trans_params = parameters["transformations"][trans["name"]]
    for param in trans_params:
        if param["name"] not in trans["params"]:
            trans["params"][param["name"]] = param["default"]

    return trans


def get_complete_clipig_source_model_config(config: dict, parameters: Optional[dict] = None) -> dict:
    if parameters is None:
        parameters = get_clipig_task_parameters()

    config = deepcopy(config)

    if not config.get("name"):
        config["name"] = "pixels"
    if not config.get("params"):
        config["params"] = {}

    params = parameters["source_models"][config["name"]]
    for param in params:
        if param["name"] not in config["params"]:
            config["params"][param["name"]] = param["default"]

    return config


