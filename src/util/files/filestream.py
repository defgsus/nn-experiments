import io
import tarfile
from functools import partial
from io import TextIOWrapper, BytesIO, StringIO, FileIO
from pathlib import Path
from typing import Optional, Union, Callable


class Filestream:
    """
    Wrapper to load/save multiple files in/from one tar file.

    It's not very efficient but enables random access buffers for reading/writing bytes and strings, e.g.

        with Filestream("file.tar", "w") as fs:
            with fs.open("content.bin", "w") as fp:
                fp.write("Plain Text")

    """

    class ByteBuffer(BytesIO):

        def __init__(self, callback: Callable[["File"], None]):
            super().__init__()
            self._callback = callback

        def __exit__(self, exc_type, exc_val, exc_tb):
            # super().__exit__(exc_type, exc_val, exc_tb)
            self.flush()
            self._callback(self)

    class StringBuffer(StringIO):

        def __init__(self, callback: Callable[["File"], None]):
            super().__init__()
            self._callback = callback

        def __exit__(self, exc_type, exc_val, exc_tb):
            # super().__exit__(exc_type, exc_val, exc_tb)
            self.flush()
            self._callback(self)

    def __init__(
            self,
            filename: Union[str, Path],
            mode: str = "r",
    ):
        self._filename = filename
        self._mode = mode
        self._file: Optional[FileIO] = None
        self._tar: Optional[tarfile.TarFile] = None

    def __enter__(self):
        self._file = open(self._filename, mode=f"{self._mode[0]}b")
        self._tar = tarfile.open(fileobj=self._file, mode=self._mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tar.close()
        self._file.close()

    def open(self, filename: Union[str, Path], mode: str = "rt", encoding: str = "utf8"):
        assert mode in ("r", "rt", "rb", "w", "wt", "wb"), f"Got {mode}"

        if mode.startswith("w") and self._mode.startswith("r"):
            raise ValueError(f"Requested mode '{mode}' but Filestream mode is '{self._mode}'")

        if mode.startswith("r"):
            tarinfo = self._tar.getmember(str(filename))

            if tarinfo.sparse is not None:
                data = b""
                for offset, size in tarinfo.sparse:
                    self._file.seek(offset)
                    data += self._file.read(size)
            else:
                self._file.seek(tarinfo.offset_data)
                data = self._file.read(tarinfo.size)

            if mode in ("r", "rt"):
                return StringIO(data.decode(encoding))

            return BytesIO(data)

        if mode == "wb":
            return self.ByteBuffer(partial(self._add_file, filename, None))
        else:
            return self.StringBuffer(partial(self._add_file, filename, encoding))

    def _add_file(self, filename: Union[str, Path], encoding: Optional[str], buffer: FileIO):
        size = buffer.tell()

        if isinstance(buffer, StringIO):
            buffer.seek(0)
            data = buffer.read().encode(encoding)
            size = len(data)
            buffer = BytesIO(data)

        buffer.seek(0)

        info = tarfile.TarInfo()
        info.size = size
        info.name = str(filename)

        self._tar.addfile(
            tarinfo=info,
            fileobj=buffer,
        )
        buffer.close()

