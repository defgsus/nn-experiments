import torch

from . import source_models


def create_source_model(config: dict, device: torch.device):
    if config["name"] not in source_models.source_models:
        raise ValueError(f"Unknown source_model '{config['name']}'")

    klass = source_models.source_models[config["name"]]

    kwargs = config["params"]

    return klass(**kwargs).to(device)



