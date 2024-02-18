import os.path
import tarfile
from pathlib import Path
from typing import Union, Optional

import yaml
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


CLIPIG_PATH = Path(__file__).resolve().parent.parent.parent


class FileDialog:

    T_Image = "image"
    T_Project = "project"

    FILE_TYPES = {
        T_Image: {
            "name": "Image",
            # first is default
            "extensions": ["png", "jpg", "bmp"],  # ...
            "default_directory": Path("~/Pictures").expanduser(),
        },
        T_Project: {
            "name": "Project",
            "extensions": ["clipig.tar"],
            "default_directory": CLIPIG_PATH / "projects"
        }
    }

    settings = {
        key: {}
        for key in FILE_TYPES.keys()
    }

    @classmethod
    def get_save_filename(
            cls,
            file_type: str,
            parent: Optional[QWidget] = None,
    ) -> Optional[str]:

        file_type_info = cls.FILE_TYPES[file_type]

        filename, used_filter = QFileDialog.getSaveFileName(
            parent=parent,
            caption=f"Save {file_type_info['name']}",
            filter=" ".join(f"*.{ext} *.{ext.upper()}" for ext in file_type_info["extensions"]),
            directory=(
                cls.settings[file_type].get("last_save_directory")
                or cls.settings[file_type].get("last_load_directory")
                or str(file_type_info["default_directory"])
            ),
        )
        if not filename:
            return None

        filename = Path(filename)
        cls.settings[file_type]["last_save_directory"] = str(filename.parent)

        if not filename.suffix:
            filename = filename.with_suffix(f".{file_type_info['extensions'][0]}")

        return str(filename)

    @classmethod
    def get_load_filename(
            cls,
            file_type: str,
            parent: Optional[QWidget] = None,
    ) -> Optional[str]:

        file_type_info = cls.FILE_TYPES[file_type]

        filename, used_filter = QFileDialog.getOpenFileName(
            parent=parent,
            caption=f"Open {file_type_info['name']}",
            filter=" ".join(f"*.{ext}" for ext in file_type_info["extensions"]) + ";;All *.*",
            directory=(
                cls.settings[file_type].get("last_load_directory")
                or cls.settings[file_type].get("last_save_directory")
                or str(file_type_info["default_directory"])
            ),
        )
        return filename or None
