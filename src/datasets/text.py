from pathlib import Path
import random
import datetime
from typing import Union, Optional, Iterable, Generator

import torch
from torch.utils.data import Dataset, IterableDataset

from .base_iterable import BaseIterableDataset
from src.util.gharchive import GHArchive


class TextSegmentIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            *text: str,
            size: int,
            stride: Union[None, int, str] = None,  # int or "random"
            encode: Optional[str] = None,  # None, "bytes"
            padding_char: str = " ",
    ):
        self._texts = text
        self._size = size
        self._stride = stride
        self._encode = encode
        self._padding_char = padding_char

    def __iter__(self):
        stride = self._stride
        if stride is None:
            stride = self._size

        for text in self._texts:
            pos = 0
            while pos < len(text):
                segment = text[pos: pos + self._size]
                if len(segment) < self._size:
                    segment = segment + self._padding_char * (self._size - len(segment))

                if stride == "random":
                    pos += random.randrange(self._size)
                elif isinstance(stride, int):
                    pos += stride
                else:
                    raise NotImplementedError(f"Invalid stride '{self._stride}'")

                if self._encode is None:
                    yield segment
                elif self._encode == "bytes":
                    yield torch.tensor(list(segment.encode()), dtype=torch.uint8)
                else:
                    raise NotImplementedError(f"Invalid encode '{self._encode}'")


class FileTextSegmentIterableDataset(BaseIterableDataset):
    """
    Base dataset must provide filenames
    """
    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
            size: int,
            stride: Union[None, int, str] = None,  # int or "random"
            encode: Optional[str] = None,
    ):
        self._dataset = dataset
        self._size = size
        self._stride = stride
        self._encode = encode

    def __iter__(self):
        stride = self._stride
        if stride is None:
            stride = self._size

        for file in self._dataset:
            try:
                text = Path(file).read_text()
            except UnicodeDecodeError:
                continue

            pos = 0
            while pos < len(text):
                segment = text[pos: pos + self._size]

                if stride == "random":
                    pos += random.randrange(self._size)
                elif isinstance(stride, int):
                    pos += stride
                else:
                    raise NotImplementedError(f"Invalid stride '{self._stride}'")

                if self._encode is None:
                    yield segment
                elif self._encode == "bytes":
                    yield torch.tensor(list(segment.encode()), dtype=torch.uint8)
                else:
                    raise NotImplementedError(f"Invalid encode '{self._encode}'")


class TextGithubEventIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            dt=datetime.datetime(2024, 12, 13, 16),
            type: Union[str, Iterable[str]] = ("commit", "comment"),
            min_text_length: Optional[int] = None,
            fixed_width: Optional[int] = None,
            stride: Union[None, int, str] = None,
            verbose: bool = True,
    ):
        super().__init__()
        self._dt = dt
        self._gha = GHArchive(verbose=verbose)
        self._type = [type] if isinstance(type, str) else set(type)
        self._min_text_length = min_text_length
        self._fixed_width = fixed_width
        self._stride = stride

    def __iter__(self) -> Generator[str, None, None]:
        for text in self._iter_texts():
            if self._min_text_length and len(text) < self._min_text_length:
                continue

            if not self._fixed_width:
                yield text
            else:
                while text:
                    if self._min_text_length and len(text) < self._min_text_length:
                        break

                    yielded_text = text
                    if self._fixed_width:
                        yielded_text = yielded_text.ljust(self._fixed_width,)

                    yield yielded_text[:self._fixed_width]

                    stride = self._stride
                    if stride is None:
                        stride = self._fixed_width
                    elif stride == "random":
                        stride = random.randrange(1, self._fixed_width)

                    text = text[stride:]

    def _iter_texts(self):
        #shown = set()
        for event in self._gha.iter_events(
                day=self._dt.date(),
                hours=self._dt.hour,
        ):
            #if event["type"] not in shown:
            #    json.dumps(event, indent=2)
            #    shown.add(event["type"])

            if event["type"] == "PushEvent":
                if "commit" in self._type:
                    if event.get("payload") and event["payload"].get("commits"):
                        for commit in event["payload"]["commits"]:
                            if commit.get("message"):
                                pass#yield commit["message"]

            elif event["type"] == "IssueCommentEvent":
                if "comment" in self._type:
                    if event.get("payload") and event["payload"].get("comment") and event["payload"]["comment"].get("body"):
                        yield event["payload"]["comment"]["body"]
                #print(json.dumps(event,indent=2))

