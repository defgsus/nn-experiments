from pathlib import Path
from typing import Union


def is_audio_file(file: Union[str, Path]) -> bool:
    ext = str(file).split(".")[-1].lower()
    return ext in _AUDIO_EXTENSIONS


_AUDIO_EXTENSIONS = {
    "wav", "mp3"
}
