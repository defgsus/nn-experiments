from pathlib import Path
from typing import Union, Generator


def iter_filenames(
        path: Union[str, Path],
        recursive: bool = False,
) -> Generator[Path, None, None]:
    """
    Yield filenames from `path`.

    :param path: str or Path, can be a
        1. filename
        2. a path
        3. a path and a glob pattern, e.g. `/path/*.txt`

    :param recursive: bool, glob files recursively

    :return: Generator of Path
    """
    path = Path(path).expanduser()

    if path.is_file():
        yield path
        return

    if path.is_dir():
        pattern = "*"
    else:
        pattern = None
        while not path.is_dir():
            pattern = path.name if pattern is None else "/".join((pattern, path.name))
            path = path.parent

    if recursive:
        iterable = path.rglob(pattern)
    else:
        iterable = path.glob(pattern)

    for filename in iterable:
        if filename.is_file():
            yield filename

