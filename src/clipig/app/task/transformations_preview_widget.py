from copy import deepcopy
from functools import partial
from typing import List, Dict

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget
from ..images import LImage, LImageSimpleWidget


class TransformationsPreviewWidget(QWidget):

    signal_run_transformation_preview = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._create_widgets()

    def set_preview_images(self, images: List[QImage]):
        limage = LImage()

        margin = 3
        x, y = 0, 0
        for idx, image in enumerate(images):
            limage.add_layer(f"example {idx + 1}", image, position=(x, y))
            x += image.width() + margin
            if idx % 5 == 4:
                x = 0
                y = y + image.height() + margin
        self.limage_widget.set_limage(limage)

    def set_running(self, running: bool):
        self.butt_preview.setDisabled(running)

    def _create_widgets(self):
        lv = QVBoxLayout(self)

        lh = QHBoxLayout()
        lv.addLayout(lh)

        self.butt_preview = QPushButton(self.tr("Preview transformations"), self)
        self.butt_preview.setToolTip(self.tr("Preview examples of current transformations"))
        self.butt_preview.clicked.connect(self.signal_run_transformation_preview)
        lh.addWidget(self.butt_preview)

        self.limage_widget = LImageSimpleWidget(self)
        lv.addWidget(self.limage_widget)

