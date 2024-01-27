import json
import gzip
from pathlib import Path
import unittest
import tempfile
from typing import Union, Tuple


class NDJson:

    def __init__(
            self,
            filename: Union[str, Path],
            mode: 'Literal["r", "w"]' = "r",
            ensure_ascii: bool = False,
            separators: Tuple[str, str] = (',', ':'),
    ):
        assert mode in "rw", mode

        self.filename = filename
        self._io = None
        self.mode = mode
        self.ensure_ascii = ensure_ascii
        self.separators = separators

    def is_zip(self) -> bool:
        return str(self.filename).lower().endswith(".gz")

    def __enter__(self):
        if self.is_zip():
            self._io = gzip.open(self.filename, f"{self.mode}t")
        else:
            self._io = open(self.filename, f"{self.mode}t")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._io is not None:
            self._io.close()
            self._io = None

    def __iter__(self):
        #if self._write:
        #    raise RuntimeError("Can not iterate NDJson(write=True)")

        if self._io is None:
            with self:
                yield from self

        else:
            for line in self._io:
                yield json.loads(line)

    def write(self, data: Union[dict, list, tuple]):
        if self.mode != "w":
            raise RuntimeError(f"Can not write to NDJson(mode={repr(self.mode)})")
        if self._io is None:
            raise RuntimeError("NdJson is not open yet")

        json.dump(data, self._io, ensure_ascii=self.ensure_ascii, separators=self.separators)
        self._io.write("\n")

    def seek(self, pos: int = 0):
        if self._io is None:
            raise RuntimeError("NdJson is not open yet")
        self._io.seek(pos)


class TestNdJson(unittest.TestCase):

    def test_read_write(self):
        with tempfile.TemporaryDirectory() as dir:
            for filename in (
                    Path(dir) / "file.ndjson",
                    Path(dir) / "file.ndjson.gz",
            ):
                with NDJson(filename, "w") as fp:
                    fp.write({"a": 1})
                    fp.write({"b": 2})

                self.assertEqual(
                    [{"a": 1}, {"b": 2}],
                    list(NDJson(filename))
                )
