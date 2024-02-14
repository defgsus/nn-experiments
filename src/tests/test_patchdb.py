import tempfile
import unittest
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as VF
import PIL.Image
from tqdm import tqdm

from src import console
from src.patchdb import PatchDB
from src.models.encoder import EncoderConv2d
from src.util.image import iter_image_patches
from tests.base import TestBase


class TestPatchDB(TestBase):

    def create_patchdb(self, shape, filename, image_filename) -> PatchDB:
        image = VF.to_tensor(PIL.Image.open(image_filename))
        patches = list(iter_image_patches(image, shape[-2:], with_pos=True))

        db = PatchDB(
            filename=filename, writeable=True,
            encoder=EncoderConv2d(shape, kernel_size=7, stride=5, channels=[4], code_size=128),
        )

        with db:
            for patch, pos in patches:
                db.add_patch(image_filename, [*pos, *shape[-2:]])

        return db

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

    def test_500_to_new_encoder(self):
        with tempfile.TemporaryDirectory() as path:
            shape = (1, 32, 32)
            db1 = self.create_patchdb(
                shape,
                Path(path) / "db.patchdb",
                self.DATA_PATH / "images/rosa.jpg",
            )
            index1 = db1.index()
            self.assertGreaterEqual(index1.size, 200)

            db2 = db1.to_new_patchdb(
                Path(path) / "db2.patchdb",
                encoder=EncoderConv2d(shape, kernel_size=15, stride=3, channels=[16], code_size=128),
            )
            index2 = db2.index()
            self.assertEqual(index1.size, index2.size)

            self.assertTensorEqual(
                index1.patches[23].patch,
                index2.patches[23].patch,
            )
            embeddings1 = [p["embedding"] for p in db1.iter_patches()]
            embeddings2 = [p["embedding"] for p in db2.iter_patches()]
            self.assertTensorNotEqual(
                torch.Tensor(embeddings1[23]),
                torch.Tensor(embeddings2[23]),
            )

    def test_900_speed(self):
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
