import tempfile
import unittest
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src import console
from src.patchdb import PatchDB
from tests.base import TestBase


class TestPatchDB(TestBase):

    def test_100_write_append(self):
        with tempfile.TemporaryDirectory() as path:

            db = PatchDB(filename=Path(path) / "db.ndjson.gz", writeable=True)
            with db:
                db._write({"a": "b"})
                db._write({"c": "d"})

            self.assertEqual(
                [{"a": "b"}, {"c": "d"}],
                list(db._read_lines())
            )

            with db:
                db._write({"e": "f"})

            self.assertEqual(
                [{"a": "b"}, {"c": "d"}, {"e": "f"}],
                list(db._read_lines())
            )

    def test_200_speed(self):
        with tempfile.TemporaryDirectory() as path:

            db = PatchDB(filename=Path(path) / "db.ndjson.gz", writeable=True)

            filename = "/bla/blub/bob.png"
            rect = (0, 1, 2, 3)
            count = 10_000

            print("PatchDB benchmark:")
            start_time = time.time()
            with db:
                for i in tqdm(range(count)):
                    embedding = torch.randn(128)
                    db.add_patch(filename, rect, embedding)
            took = time.time() - start_time

            print(f"{count/took:.3f} writes per second")
            size = db.size_bytes()
            print(f"filesize: {size:,} bytes, {size//count} bytes per patch ({embedding.shape[-1]} dim)")

            start_time = time.time()
            list(db.iter_patches())
            took = time.time() - start_time

            print(f"{count/took:.3f} reads per second")
            
            # --- index ---

            start_time = time.time()
            index = db.index()
            self.assertEqual(count, index.size)
            took = time.time() - start_time

            print(f"building index: {count/took:.3f} patches per second ")

            print(index.similar_patches(torch.randn(128), 10))
