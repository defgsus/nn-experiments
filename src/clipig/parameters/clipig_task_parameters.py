import json
from pathlib import Path
from copy import deepcopy
from typing import Optional, List

import yaml


with (Path(__file__).resolve().parent / "clipig_task_parameters.yaml").open() as fp:
    _BASE_PARAMETERS = yaml.safe_load(fp)


def get_clipig_task_parameters():
    """
    Combine the clipig_task_parameters.yaml params and
    the runtime generated params from transformation and source-model classes
    """
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

    patch_values_with_parameters(config, parameters["base"])

    config["source_model"] = get_complete_clipig_source_model_config(
        config.get("source_model") or {},
        parameters=parameters,
    )

    if "targets" not in config:
        config["targets"] = []

    for target in config["targets"]:
        patch_values_with_parameters(target, parameters["target"])

        # -- update prompt to new format --
        if "target_features" not in target:
            target["target_features"] = [
                {
                    "text": target["prompt"],
                    "weight": 1.
                }
            ]
            if target.get("negative_prompt"):
                target["target_features"].append({
                    "text": target["negative_prompt"],
                    "weight": -1.
                })

        target.pop("prompt", None)
        target.pop("negative_prompt", None)

        for feature in target["target_features"]:
            patch_values_with_parameters(feature, parameters["target_feature"])

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

    patch_values_with_parameters(
        trans["params"],
        parameters["transformations"][trans["name"]]
    )

    return trans


def get_complete_clipig_source_model_config(config: dict, parameters: Optional[dict] = None) -> dict:
    if parameters is None:
        parameters = get_clipig_task_parameters()

    config = deepcopy(config)

    if not config.get("name"):
        config["name"] = "pixels"
    if not config.get("params"):
        config["params"] = {}

    patch_values_with_parameters(
        config["params"],
        parameters["source_models"][config["name"]]
    )

    return config


def patch_values_with_parameters(values: dict, parameters: List[dict]):
    for param in parameters:
        if param["name"] not in values:
            values[param["name"]] = param["default"]

