from pathlib import Path
import random
import datetime
from typing import Union, Optional, Iterable, Generator, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from .base_iterable import BaseIterableDataset
from src.util.gharchive import GHArchive


def make_compact_whitespace(text: str):
    compact_text = []
    was_whitespace = False
    for ch in text:
        if ch.isspace():
            if not was_whitespace:
                compact_text.append(" ")
            was_whitespace = True
        else:
            compact_text.append(ch)
            was_whitespace = False

    return "".join(compact_text)


class TextSegmentIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            *text: str,
            size: Union[int, Tuple[int, int]],
            batch_size: Optional[int] = None,
            compact_whitespace: bool = True,
            stride: Union[None, int, str] = None,  # int or "random"
            encode: Optional[str] = None,  # None, "bytes"
            padding_char: str = " ",
            seed: Optional[int] = None,
    ):
        assert isinstance(size, int) or batch_size is not None, "Must provide batch_size when size is tuple"
        if compact_whitespace:
            text = [make_compact_whitespace(t) for t in text]
        self._texts = text
        self._size = size
        self._batch_size = batch_size
        self._stride = stride
        self._encode = encode
        self._padding_char = padding_char
        self._rng = random if seed is None else random.Random(seed)

    def __iter__(self):
        stride = self._stride
        if stride is None:
            stride = self._size

        size = self._size
        if not isinstance(size, int):
            size = self._rng.randrange(*size)

        counter = 0
        for text in self._texts:
            pos = 0
            while pos < len(text):
                segment = text[pos: pos + size]
                if len(segment) < size:
                    segment = segment + self._padding_char * (size - len(segment))

                if stride == "random":
                    pos += self._rng.randrange(size)
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

                counter += 1
                if not isinstance(self._size, int):
                    if counter >= self._batch_size:
                        counter = 0
                        size = self._rng.randrange(*self._size)


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


class TextWiki9SegmentIterableDataset(BaseIterableDataset):
    def __init__(
            self,
            size: Union[int, Tuple[int, int]],
            batch_size: Optional[int] = None,
            compact_whitespace: bool = True,
            stride: Union[None, int, str] = None,  # int or "random"
            encode: Optional[str] = None,  # None, "bytes"
            padding_char: str = " ",
            seed: Optional[int] = None,
    ):
        from torchtext.datasets import EnWik9

        assert isinstance(size, int) or batch_size is not None, "Must provide batch_size when size is tuple"
        self._ds = EnWik9()
        self._size = size
        self._batch_size = batch_size
        self._compact_whitespace = compact_whitespace
        self._stride = stride
        self._encode = encode
        self._padding_char = padding_char
        self._rng = random if seed is None else random.Random(seed)

    def __iter__(self):
        stride = self._stride
        if stride is None:
            stride = self._size

        size = self._size
        if not isinstance(size, int):
            size = self._rng.randrange(*size)

        def _iter_text():
            lines = []
            for line in self._ds:
                lines.append(line)
                if len(lines) >= 100:
                    yield "\n".join(lines)
                    lines.clear()
            if lines:
                yield "\n".join(lines)

        counter = 0
        for text in _iter_text():
            if self._compact_whitespace:
                text = make_compact_whitespace(text)
            pos = 0
            while pos < len(text):
                segment = text[pos: pos + size]
                if len(segment) < size:
                    segment = segment + self._padding_char * (size - len(segment))

                if stride == "random":
                    pos += self._rng.randrange(size)
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

                counter += 1
                if not isinstance(self._size, int):
                    if counter >= self._batch_size:
                        counter = 0
                        size = self._rng.randrange(*self._size)
