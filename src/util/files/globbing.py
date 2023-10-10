from pathlib import Path
from typing import Union, Generator, Iterable


def iter_filenames(
        path: Union[str, Path, Iterable[Union[str, Path]]],
        recursive: bool = False,
) -> Generator[Path, None, None]:
    """
    Yield filenames from `path`.

    :param path: str or Path, can be a
        1. filename
        2. a path
        3. a path and a glob pattern, e.g. `/path/*.txt`
        4. an iterable of any of the above

    :param recursive: bool, glob files recursively

    :return: Generator of Path
    """
    if not isinstance(path, (str, Path)):
        for p in path:
            yield from iter_filenames(p, recursive=recursive)
        return

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

