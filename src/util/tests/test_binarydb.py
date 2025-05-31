import tempfile
from pathlib import Path

import torch

from src.tests.base import *
from src.util.binarydb import BinaryDB


class TestBinaryDB(TestBase):

    def test_100_get_set(self):
        with tempfile.TemporaryDirectory("nn-unittest") as dir:
            db = BinaryDB(Path(dir) / "db.sqlite")
            with db:
                db.store("a", b"a")
                db.store("b", b"bb", {"meta": "data"})

            with db:
                self.assertEqual((b"a", None), db.get("a"))
                self.assertEqual((b"bb", {"meta": "data"}), db.get("b"))
                self.assertEqual(None, db.get("c"))

                self.assertTrue(db.has("a"))
                self.assertTrue(db.has("b"))
                self.assertFalse(db.has("c"))


    def test_200_iter(self):
        self.maxDiff = 10_000
        with tempfile.TemporaryDirectory("nn-unittest") as dir:
            db = BinaryDB(Path(dir) / "db.sqlite")
            with db:
                expected_data = []
                for i in range(23):
                    expected_data.append((f"{i:04}", b"i" * i, {"i": i}))
                    db.store(*expected_data[-1])

                rows = list(db.iter())
                self.assertEqual(
                    expected_data,
                    rows
                )
