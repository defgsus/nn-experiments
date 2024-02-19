import math
import unittest
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.clipig.transformations.value_trans import *


class TestTransformations(unittest.TestCase):

    def test_denoising(self):
        trans = Denoising(model="denoiser-conv-32x32-750k", mix=1.)

        trans(torch.ones(3, 301, 302))
