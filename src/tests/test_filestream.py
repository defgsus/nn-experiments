import math
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Tuple

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from src.util.files import Filestream
from src.tests.base import TestBase

class TestFilestream(TestBase):

    def test_100_filestream_file(self):
        #qimage = QImage(QSize(100, 100), QImage.Format.Format_ARGB32)
        qimage = QImage(str(self.DATA_PATH / "images" / "rosa.jpg"))

        with tempfile.TemporaryDirectory("clipig") as dir:
            dir = Path(dir)

            with Filestream(dir / "archive.tar", "w") as fs:

                with fs.open("content.bin", "wb") as fp:
                    fp.write(b"data" * 100_000)

                with fs.open("content.txt", "w") as fp:
                    fp.write("ünicödä" * 100_000)

                fs.write_qimage("content.png", qimage)

            with Filestream(dir / "archive.tar") as fs:

                with self.assertRaises(ValueError):
                    fs.open("content.bin", "w")

                with fs.open("content.bin", "rb") as fp:
                    self.assertEqual(b"data" * 100_000, fp.read())

                with fs.open("content.txt") as fp:
                    self.assertEqual("ünicödä" * 100_000, fp.read())

                qimage2 = fs.read_qimage("content.png")

                self.assertEqual(qimage.size(), qimage2.size())
                self.assertEqual(qimage.sizeInBytes(), qimage2.sizeInBytes())

    def test_200_filestream_buffer(self):
        #qimage = QImage(QSize(100, 100), QImage.Format.Format_ARGB32)
        qimage = QImage(str(self.DATA_PATH / "images" / "rosa.jpg"))

        buffer = BytesIO()
        with Filestream(buffer, "w") as fs:

            with fs.open("content.bin", "wb") as fp:
                fp.write(b"data" * 100_000)

            with fs.open("content.txt", "w") as fp:
                fp.write("ünicödä" * 100_000)

            fs.write_yaml("some.yml", {"a": {"b": "c"}})

            fs.write_qimage("content.png", qimage)

        buffer.seek(0)
        with Filestream(buffer) as fs:

            with self.assertRaises(ValueError):
                fs.open("content.bin", "w")

            with fs.open("content.bin", "rb") as fp:
                self.assertEqual(b"data" * 100_000, fp.read())

            with fs.open("content.txt") as fp:
                self.assertEqual("ünicödä" * 100_000, fp.read())

            self.assertEqual({"a": {"b": "c"}}, fs.read_yaml("some.yml"))

            qimage2 = fs.read_qimage("content.png")

            self.assertEqual(qimage.size(), qimage2.size())
            self.assertEqual(qimage.sizeInBytes(), qimage2.sizeInBytes())
