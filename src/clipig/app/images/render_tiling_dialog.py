import os.path
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Tuple, Any

import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..parameters import ParameterWidget, ParametersWidget
from .limage import LImage, LImageLayer
from .limage_simple_widget import LImageSimpleWidget
from .render_tiling import TilingTemplateRenderer
from ..util import torch_to_qimage


class RenderTilingDialog(QDialog):

    def __init__(
            self,
            limage: LImage,
            project: "ProjectWidget",
    ):
        from ..project import ProjectWidget
        super().__init__(project)
        self.project: ProjectWidget = project
        self.limage = limage
        self.rendered_limage = LImage()
        x, y = self.limage.tiling.map_size
        self.setWindowTitle(f"Render tiling template ({x}, {y})")

        self._create_widgets()
        self.set_config(
            self.project.get_dialog_settings("tile_template_renderer")
        )

    def _create_widgets(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(20, 20, 20, 20)

        parameters = TilingTemplateRenderer.PARAMETERS.copy()
        #s = self.limage.size()
        #list(filter(lambda p: p["name"] == "resolution", parameters))[0]["default"] = [s.width(), s.height()]
        list(filter(lambda p: p["name"] == "tile_resolution", parameters))[0]["default"] = self.limage.tiling.tile_size
        list(filter(lambda p: p["name"] == "template_size", parameters))[0]["default"] = self.limage.tiling.map_size

        lh = QHBoxLayout()
        lv.addLayout(lh)

        self.parameters_widget = ParametersWidget(self, parameters=parameters)
        lh.addWidget(self.parameters_widget)

        lv2 = QVBoxLayout()
        lh.addLayout(lv2)

        self.render_button = QPushButton(self.tr("Render"), self)
        self.render_button.clicked.connect(self._render)
        lv2.addWidget(self.render_button)

        self.image_widget = LImageSimpleWidget(project=self.project)
        self.image_widget.set_limage(self.rendered_limage)
        lv2.addWidget(self.image_widget)

        lv.addSpacing(20)

        self.layer_name_widget = ParameterWidget({
            "name": "layer_name",
            "type": "str",
            "default": "template",
        }, self)
        lv.addWidget(self.layer_name_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self
        )
        buttons.accepted.connect(self._accept_clicked)
        buttons.rejected.connect(self.reject)
        lv.addWidget(buttons)

    def get_config(self):
        return {
            **self.parameters_widget.get_values(),
            "layer_name": self.layer_name_widget.get_value(),
        }

    def set_config(self, values: dict):
        self.parameters_widget.set_values(values, emit=False)
        if values.get("layer_name"):
            self.layer_name_widget.set_value(values["layer_name"])

    def _render(self):
        config = self.get_config()
        array = TilingTemplateRenderer.render(
            tiling=self.limage.tiling,
            config=config,
        )
        image = torch_to_qimage(torch.from_numpy(array).permute(2, 0, 1))

        layer_name = config["layer_name"]
        layer = self.rendered_limage.get_layer(layer_name)
        if layer is None:
            layer = self.rendered_limage.add_layer(layer_name)
        layer.set_image(image)

    def _accept_clicked(self):
        self.accept()

    @classmethod
    def run_dialog(
            cls,
            limage: LImage,
            project: "ProjectWidget",
    ) -> Optional[LImageLayer]:
        if not limage.tiling:
            return None

        dialog = cls(
            limage=limage,
            project=project,
        )
        if dialog.exec_():
            dialog._render()
            dialog.project.set_dialog_settings("tile_template_renderer", dialog.get_config())
            return dialog.rendered_limage.get_layer(dialog.get_config()["layer_name"])
