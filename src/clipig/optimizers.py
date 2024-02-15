import time
from copy import deepcopy
from typing import Tuple, Optional, Union, Iterable, Generator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF



# parameters are defined in `clipig_task_parameters.yaml`
def create_optimizer(params: Iterable[nn.Parameter], config: dict):
    config = deepcopy(config)
    config["lr"] = config.pop("learnrate")

    name = config.pop("optimizer")
    klass = getattr(torch.optim, name)

    return klass(params, **config)

