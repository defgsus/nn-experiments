import tempfile

import torch

from src.tests.base import *
from src.util import TensorStorage


class TestTensorStorage(TestBase):

    def test_100_test_num_items(self):
        with tempfile.TemporaryDirectory("nn-test") as root:
            root = Path(root)

            storage = TensorStorage(
                filename_part=root / "batch",
                max_items=10,
                num_zero_padding_digits=2,
            )

            tensors = torch.rand(35, 100)
            for t in tensors:
                storage.add(t)
            storage.store_buffer()

            files = sorted(f.name for f in root.glob("*"))
            self.assertEqual(
                [
                    "batch-00.pt",
                    "batch-01.pt",
                    "batch-02.pt",
                    "batch-03.pt",
                ],
                files
            )

            self.assertTensorEqual(tensors[ 0:10], torch.load(root / "batch-00.pt"))
            self.assertTensorEqual(tensors[10:20], torch.load(root / "batch-01.pt"))
            self.assertTensorEqual(tensors[20:30], torch.load(root / "batch-02.pt"))
            self.assertTensorEqual(tensors[30:35], torch.load(root / "batch-03.pt"))

    def test_200_test_num_bytes(self):
        with tempfile.TemporaryDirectory("nn-test") as root:
            root = Path(root)

            storage = TensorStorage(
                filename_part=root / "batch",
                max_bytes=10_000,
                num_zero_padding_digits=2,
            )

            tensors = torch.rand(5, 1000, dtype=torch.float64)
            for t in tensors:
                storage.add(t)
            storage.store_buffer()

            files = sorted(f.name for f in root.glob("*"))
            self.assertEqual(
                [
                    "batch-00.pt",
                    "batch-01.pt",
                    "batch-02.pt",
                    "batch-03.pt",
                    "batch-04.pt",
                ],
                files
            )

            self.assertTensorEqual(tensors[0:1], torch.load(root / "batch-00.pt"))
            self.assertTensorEqual(tensors[1:2], torch.load(root / "batch-01.pt"))
            self.assertTensorEqual(tensors[2:3], torch.load(root / "batch-02.pt"))
            self.assertTensorEqual(tensors[3:4], torch.load(root / "batch-03.pt"))
            self.assertTensorEqual(tensors[4:5], torch.load(root / "batch-04.pt"))

            for file in root.glob("*.pt"):
                self.assertLessEqual(file.stat().st_size, 10_000, f"for file {file.name}")

