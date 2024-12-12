from pathlib import Path
import unittest
from typing import Union

import torch
import numpy as np


class TestBase(unittest.TestCase):

    DATA_PATH = Path(__file__).resolve().parent / "data"

    def assertTensorEqual(
            self,
            expected: Union[list, torch.Tensor],
            real: Union[list, torch.Tensor],
            msg: str = "",
            places: int = 4,
    ):
        if isinstance(expected, list):
            expected = torch.Tensor(expected)

        if isinstance(real, list):
            real = torch.Tensor(real)

        if "int" not in str(expected.dtype):
            expected = expected.round(decimals=places)
        expected = expected.tolist()

        if "int" not in str(real.dtype):
            real = real.round(decimals=places)
        real = real.tolist()

        self.assertEqual(expected, real, f"\nExpected:\n{expected}\nGot:\n{real}" + (f", {msg}" if msg else ""))

    def assertNumpyEqual(
            self,
            expected: Union[list, np.ndarray],
            real: Union[list, np.ndarray],
            msg: str = "",
            places: int = 4,
    ):
        if isinstance(expected, list):
            expected = np.array(expected)

        if isinstance(real, list):
            real = np.array(real)

        if "int" not in str(expected.dtype):
            expected = np.round(expected, decimals=places)
        expected = expected.tolist()

        if "int" not in str(real.dtype):
            real = np.round(real, decimals=places)
        real = real.tolist()

        self.assertEqual(expected, real, f"\nExpected:\n{expected}\nGot:\n{real}" + (f", {msg}" if msg else ""))

    def assertTensorNotEqual(self, expected: Union[list, torch.Tensor], real: Union[list, torch.Tensor]):
        if isinstance(expected, list):
            expected = torch.Tensor(expected).tolist()

        if isinstance(expected, torch.Tensor):
            expected = expected.tolist()

        if isinstance(real, list):
            real = torch.Tensor(real).tolist()

        if isinstance(real, torch.Tensor):
            real = real.tolist()

        self.assertNotEqual(expected, real, f"\nExpected unequal to:\n{expected}\nGot:\n{real}")
