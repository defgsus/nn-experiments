import io
import tarfile
import datetime
from functools import partial
from io import TextIOWrapper, BytesIO, StringIO, FileIO
from pathlib import Path
from typing import Optional, Union, Callable, List, IO

import yaml


class Filestream:
    """
    Wrapper to load/save multiple files in/from one tar file.

    It's not very efficient but enables random access buffers for reading/writing bytes and strings, e.g.

        with Filestream("file.tar", "w") as fs:
            with fs.open("content.bin", "w") as fp:
                fp.write("Plain Text")

    """

    def __init__(
            self,
            file: Union[str, Path, IO[bytes]],
            mode: str = "r",
    ):
        self._file_arg = file
        self._is_filename = isinstance(self._file_arg, (str, Path))
        self._mode = mode
        self._file: Optional[FileIO] = None
        self._tar: Optional[tarfile.TarFile] = None

    def __enter__(self):
        if self._is_filename:
            self._file = open(self._file_arg, mode=f"{self._mode[0]}b")
        else:
            self._file = self._file_arg
        self._tar = tarfile.open(fileobj=self._file, mode=self._mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tar.close()
        if self._is_filename:
            self._file.close()

    def open(self, filename: Union[str, Path], mode: str = "rt", encoding: str = "utf8"):
        """
        Open a file inside the tar for reading or writing.

        When opening for writing: the returned buffer must be closed at which point
        the contents are written to the tarfile.

        :param filename: str|Path, the local filename (inside tar)
        :param mode: str, can be "r", "rt", "rb", "w", "wt", or "wb"
        :param encoding: str, encoding passed t TextIOWrapper
        :return: a matching subclass instances of `BufferedIO`
        """
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

        buffer = _ByteBuffer(partial(self._write_buffer_callback, filename))
        if mode in ("w", "wt"):
            buffer = TextIOWrapper(buffer, encoding=encoding)
        return buffer

    def filenames(self) -> List[str]:
        return [
            info.name
            for info in self._tar.getmembers()
            if info.isfile()
        ]

    # --- convenience functions ----

    def write_bytes(self, filename: Union[str, Path], data: bytes):
        with self.open(filename, "wb") as fp:
            fp.write(data)

    def write_text(self, filename: Union[str, Path], data: str, encoding: str = "utf8"):
        with self.open(filename, "wt", encoding=encoding) as fp:
            fp.write(data)

    def read_bytes(self, filename: Union[str, Path]) -> bytes:
        with self.open(filename, "rb") as fp:
            return fp.read()

    def read_text(self, filename: Union[str, Path], encoding: str = "utf8") -> str:
        with self.open(filename, "rt", encoding=encoding) as fp:
            return fp.read()

    def write_yaml(self, filename: Union[str, Path], data: Union[list, dict]):
        with self.open(filename, "wt") as fp:
            yaml.safe_dump(data, fp)

    def read_yaml(self, filename: Union[str, Path]) -> Union[list, dict]:
        with self.open(filename, "rt") as fp:
            return yaml.safe_load(fp)

    def write_qimage(self, filename: Union[str, Path], qimage, format: Optional[str] = None):
        from PyQt5.QtCore import QBuffer

        device = QBuffer()
        qimage.save(device, format or Path(filename).suffix[1:])
        self.write_bytes(filename, device.data().data())

    def read_qimage(self, filename: Union[str, Path], format: Optional[str] = None):
        from PyQt5.QtCore import QBuffer, QByteArray
        from PyQt5.QtGui import QImage

        data = self.read_bytes(filename)
        buffer = QBuffer()
        buffer.setData(QByteArray(data))
        buffer.open(QBuffer.ReadOnly)
        image = QImage()
        image.load(buffer, format or Path(filename).suffix[1:])
        buffer.close()
        return image

    def _write_buffer_callback(self, filename: Union[str, Path], buffer: "_ByteBuffer"):
        if self._tar is None:
            raise RuntimeError(f"Write on unopened Filestream {self}")
        size = buffer.tell()
        buffer.seek(0)

        info = tarfile.TarInfo()
        info.size = size
        info.name = str(filename)
        info.mtime = int(datetime.datetime.now().timestamp())

        self._tar.addfile(
            tarinfo=info,
            fileobj=buffer,
        )
        buffer.super_close()


# this wrapper is returned from `filestream.open(mode="w?")`
#   instead of closing the buffer, it calls the callback function
#   which extracts the content and then closes it for real

class _ByteBuffer(BytesIO):

    def __init__(self, callback: Callable[["_ByteBuffer"], None]):
        super().__init__()
        self._callback = callback

    def close(self):
        self._callback(self)

    def super_close(self):
        super().close()
