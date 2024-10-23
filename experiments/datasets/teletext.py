import math
import random
import fnmatch
import gzip
from pathlib import Path
from functools import partial
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset

from src.datasets import *
from src.util.image import *
from src.util.files import iter_ndjson


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "datasets"


class TeletextIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            filename: Union[str, Path] = DATASET_PATH / "teletext.ndjson.gz",
            total: Optional[int] = 15_505_000,
    ):
        super().__init__()
        self._filename = Path(filename)
        self._total = total

    def __len__(self):
        if self._total is not None:
            return self._total
        raise AttributeError(f"No __len__ defined for {self.__class__}")

    def __iter__(self):
        for entry in iter_ndjson(self._filename):
            text = entry.pop("text")
            yield text, entry


class TeletextPixelIterableDataset(TextToPixelIterableDataset):

    def __init__(
            self,
            filename: Union[str, Path] = DATASET_PATH / "teletext.ndjson.gz",
            total: Optional[int] = 15_505_000,
            font_shape: Tuple[int, int] = (8, 8),
            font_file: Union[str, Path] = Path("~/.local/share/fonts/unscii-8.ttf").expanduser(),
    ):
        super().__init__(
            dataset=TeletextIterableDataset(filename=filename, total=total),
            screen_size=(24, 40),
            font_shape=font_shape,
            font_file=font_file,
        )

    def __len__(self):
        return len(self._dataset)



class TeletextMatrixIterableDataset(BaseIterableDataset):

    CHARACTERS = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz~\x80\x81\x82\x84\x87\x89\x93\x96\x98\x99\x9c\x9f\xa0¡£¥¦§¨©ª«\xad®¯°±²³·¸º»¼½¾ÀÁÂÄÅÇÉËÓÖØÙÚÜßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýÿćĈČčēěğıłńňřŚśŝŞşŠšž–‘’“”€█▌▐\U0001fb00\U0001fb01\U0001fb02\U0001fb03\U0001fb04\U0001fb05\U0001fb06\U0001fb07\U0001fb08\U0001fb09\U0001fb0a\U0001fb0b\U0001fb0c\U0001fb0d\U0001fb0e\U0001fb0f\U0001fb10\U0001fb11\U0001fb12\U0001fb13\U0001fb14\U0001fb15\U0001fb16\U0001fb17\U0001fb18\U0001fb19\U0001fb1a\U0001fb1b\U0001fb1c\U0001fb1d\U0001fb1e\U0001fb1f\U0001fb20\U0001fb21\U0001fb22\U0001fb23\U0001fb24\U0001fb25\U0001fb26\U0001fb27\U0001fb28\U0001fb29\U0001fb2a\U0001fb2b\U0001fb2c\U0001fb2d\U0001fb2e\U0001fb2f\U0001fb30\U0001fb31\U0001fb32\U0001fb33\U0001fb34'

    DIM = len(CHARACTERS)
    assert DIM <= 256, DIM

    CHARACTER_TO_INDEX = {
        c: i
        for i, c in enumerate(CHARACTERS)
    }

    def __init__(
            self,
            filename: Union[str, Path] = DATASET_PATH / "teletext.ndjson.gz",
            total: Optional[int] = 15_505_000,
            meta: bool = False,
    ):
        super().__init__()
        self._filename = Path(filename)
        self._meta = meta
        self._total = total

    def __len__(self):
        if self._total is not None:
            return self._total
        raise AttributeError(f"No __len__ defined for {self.__class__}")

    def __iter__(self):
        for entry in iter_ndjson(self._filename):
            text = entry.pop("text")
            matrix = self.text_to_matrix(text)
            if self._meta:
                yield matrix, entry
            else:
                yield matrix

    @classmethod
    def text_to_matrix(cls, text):
        matrix = torch.zeros(cls.DIM, 20, 40)
        for y, line in enumerate(text.splitlines()):
            if y >= 20:
                break
            for x, ch in enumerate(line):
                idx = cls.CHARACTER_TO_INDEX.get(ch, 0)
                if x >= 40:
                    break
                matrix[idx, y, x] = 1.
        return matrix
        #yield text, entry

    @classmethod
    def matrix_to_text(cls, matrix):
        arg_max = matrix.argmax(dim=0)
        text_lines = [[" "] * 40 for _ in range(20)]
        for y in range(20):
            for x in range(40):
                text_lines[y][x] = cls.CHARACTERS[arg_max[y, x]]

        return "\n".join("".join(line) for line in text_lines)

