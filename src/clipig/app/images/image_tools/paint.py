from .base import *
from .brushes import Brush


class PaintTool(ImageToolBase):

    NAME = "paint"
    PARAMS = [
        *Brush.PARAMS,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def config_changed(self):
        self.brush = Brush(
            type=self.config["type"],
            size=self.config["size"],
            color=self.config["color"],
            alpha=self.config["alpha"],
        )

    def mouse_event(self, event: MouseEvent):
        if event.button == Qt.LeftButton:
            if event.type in (MouseEvent.Drag, MouseEvent.Press):

                if self.layer.active:
                    with self.layer.image_painter() as painter:

                        self.brush.apply(self.limage, painter, event.pos)

