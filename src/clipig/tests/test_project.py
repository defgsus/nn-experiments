import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from src.clipig.clipig_worker import ClipigWorker
from src.clipig.clipig_task import ClipigTask
from src.clipig.app.project import ProjectWidget
from src.clipig.app.models import PresetModel
from src.clipig.app.images import LImage
from src.util.files import Filestream


class TestProject(unittest.TestCase):

    def setUp(self):
        self.app = QApplication(sys.argv)

    def create_limage(self) -> LImage:
        image = LImage()
        l = image.add_layer(name="A", image=QImage(QSize(10, 20), QImage.Format.Format_ARGB32))
        l.set_active(False)
        l = image.add_layer(name="B", image=QImage(QSize(200, 100), QImage.Format.Format_Mono))
        l.set_selected()
        return image

    def test_project_io(self):
        clipig = ClipigWorker()
        preset_model = PresetModel()

        project = ProjectWidget(clipig=clipig, preset_model=preset_model)
        task = project.slot_new_task()

        task.image_widget.set_limage(self.create_limage())

        with tempfile.TemporaryDirectory("clipig") as dir:
            project_filename = Path(dir) / "project.clipig.tar"

            project.save_project(project_filename)

            with Filestream(project_filename) as fs:
                self.assertEqual(
                    [
                        "config.yaml",
                        "task_00/config.yaml",
                        "task_00/limage/config.yaml",
                        "task_00/limage/layer_00.png",
                        "task_00/limage/layer_01.png",
                    ],
                    sorted(fs.filenames())
                )

            project2 = ProjectWidget(clipig=clipig, preset_model=preset_model)
            project2.load_project(project_filename)
