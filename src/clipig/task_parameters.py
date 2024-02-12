import json
from pathlib import Path
from copy import deepcopy

import yaml


with (Path(__file__).resolve().parent / "task_parameters.yaml").open() as fp:
    _BASE_PARAMETERS = yaml.safe_load(fp)


def get_task_parameters():
    from . import transformations
    from . import source_models

    parameters = deepcopy(_BASE_PARAMETERS)

    parameters["transformations"] = {}
    for klass in transformations.transformations.values():
        parameters["transformations"][klass.NAME] = deepcopy(klass.PARAMS)

    parameters["source_models"] = {}
    for klass in source_models.source_models.values():
        parameters["source_models"][klass.NAME] = deepcopy(klass.PARAMS)

    # print(json.dumps(parameters, indent=2))
    return parameters


def get_complete_task_config(config: dict) -> dict:
    """
    Add all default values to the config dict.

    Returns new instance
    """
    config = deepcopy(config)
    parameters = get_task_parameters()

    for param in parameters["base"]:
        if param["name"] not in config:
            config[param["name"]] = param["default"]

    if "targets" not in config:
        config["targets"] = []

    for target in config["targets"]:
        for param in parameters["target"]:
            if param["name"] not in target:
                target[param["name"]] = param["default"]

        if "transformations" not in target:
            target["transformations"] = []

        for trans in target["transformations"]:
            trans_params = parameters["transformations"][trans["name"]]
            for param in trans_params:
                if param["name"] not in trans["params"]:
                    trans["params"][param["name"]] = param["default"]

    return config


def get_complete_transformation_config(trans: dict) -> dict:
    trans = deepcopy(trans)
    parameters = get_task_parameters()

    trans_params = parameters["transformations"][trans["name"]]
    for param in trans_params:
        if param["name"] not in trans["params"]:
            trans["params"][param["name"]] = param["default"]

    return trans


