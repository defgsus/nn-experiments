import math
import unittest
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.clipig.transformations.value_trans import *


class TestTransformations(unittest.TestCase):

    def test_denoising(self):
        def_model = next(filter(lambda p: p["name"] == "model", Denoising.PARAMS))["default"]
        trans = Denoising(model=def_model, mix=1.)

        trans(torch.ones(3, 301, 302))
