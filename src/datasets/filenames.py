from pathlib import Path
from typing import Union, Iterable, Optional

import fnmatch
import glob

from .base_dataset import BaseDataset


class FilenameDataset(BaseDataset):
    """
    Use like

        dsf = FilenameDataset(
            "~/code/",
            include="*.py",
            exclude="*/env/*",
            recursive=True,
        )

    """
    def __init__(
            self,
            root: Union[str, Path],
            include: Union[None, str, Iterable[str]] = None,
            exclude: Union[None, str, Iterable[str]] = None,
            recursive: bool = False,
            max_files: Optional[int] = None,
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        if include is None:
            self.include = None
        elif isinstance(include, str):
            self.include = [include]
        else:
            self.include = list(include)
        if exclude is None:
            self.exclude = None
        elif isinstance(exclude, str):
            self.exclude = [exclude]
        else:
            self.exclude = list(exclude)

        self.recursive = recursive
        self.max_files = max_files
        self._filenames = None

    def __len__(self):
        self._get_filenames()
        return len(self._filenames)

    def __getitem__(self, i):
        self._get_filenames()
        return self._filenames[i]

    def _is_valid(self, filename: str) -> bool:
        if self.include:
            for pattern in self.include:
                if not fnmatch.fnmatch(filename, pattern):
                    return False

        if self.exclude:
            for pattern in self.exclude:
                if fnmatch.fnmatch(filename, pattern):
                    return False
        return True

    def _get_filenames(self):
        if self._filenames is None:

            if self.root.is_file():
                self._filenames = [str(self.root)]

            else:
                glob_path = self.root
                if self.recursive:
                    glob_path /= "**/*"
                else:
                    glob_path /= "*"

                self._filenames = []
                for filename in glob.glob(str(glob_path), recursive=self.recursive):
                    if self._is_valid(filename):
                        self._filenames.append(filename)
                        if self.max_files and len(self._filenames) >= self.max_files:
                            break

                self._filenames.sort()
