import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

from src.util.files import Filestream


class TestFilestream(unittest.TestCase):

    def test_filestream(self):
        with tempfile.TemporaryDirectory("clipig") as dir:
            dir = Path(dir)

            with Filestream(dir / "archive.tar", "w") as fs:

                with fs.open("content.bin", "wb") as fp:
                    fp.write(b"data" * 100_000)

                with fs.open("content.txt", "w") as fp:
                    fp.write("ünicödä" * 100_000)

            with Filestream(dir / "archive.tar") as fs:

                with self.assertRaises(ValueError):
                    fs.open("content.bin", "w")

                with fs.open("content.bin", "rb") as fp:
                    self.assertEqual(b"data" * 100_000, fp.read())

                with fs.open("content.txt") as fp:
                    self.assertEqual("ünicödä" * 100_000, fp.read())
