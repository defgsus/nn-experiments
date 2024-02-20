from tqdm import tqdm

from src.tests.base import *
from src.datasets import RpgTileIterableDataset


class TestRpgDataset(TestBase):

    def test_100_download(self):
        for _ in tqdm(RpgTileIterableDataset(verbose=True), desc="reading rpg dataset"):
            pass
