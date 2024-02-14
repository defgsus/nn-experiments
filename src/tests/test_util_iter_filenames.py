import torch

from tests.base import *
from src.util.files import *


class TestUtilIterFilenames(TestBase):

    def test_100_file(self):
        self.assertEqual(
            ["images/jim.jpg"],
            sorted(
                str(f.relative_to(self.DATA_PATH))
                for f in iter_filenames(self.DATA_PATH / "images/jim.jpg")
            )
        )

    def test_200_directory(self):
        self.assertEqual(
            ["images/camera.jpg",
             "images/capitalism.jpg",
             "images/hornauer.jpg",
             "images/jim.jpg",
             "images/max.jpeg",
             "images/rosa.jpg",
             "images/sub/MANSON.JPG",
            ],
            sorted(
                str(f.relative_to(self.DATA_PATH))
                for f in iter_filenames(self.DATA_PATH / "images", recursive=True)
            )
        )

    def test_300_wildcard(self):
        self.assertEqual(
            ["images/rosa.jpg"],
            sorted(
                str(f.relative_to(self.DATA_PATH))
                for f in iter_filenames(self.DATA_PATH / "rosa.*", recursive=True)
            )
        )
