from pathlib import Path
import unittest
from typing import Union

import torch


class TestBase(unittest.TestCase):

    DATA_PATH = Path(__file__).resolve().parent / "data"

    def assertTensorEqual(self, expected: Union[list, torch.Tensor], real: Union[list, torch.Tensor]):
        if isinstance(expected, list):
            expected = torch.Tensor(expected).tolist()

        if isinstance(expected, torch.Tensor):
            expected = expected.tolist()

        if isinstance(real, list):
            real = torch.Tensor(real).tolist()

        if isinstance(real, torch.Tensor):
            real = real.tolist()

        self.assertEqual(expected, real, f"\nExpected:\n{expected}\nGot:\n{real}")
