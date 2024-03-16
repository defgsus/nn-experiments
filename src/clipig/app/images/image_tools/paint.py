from functools import partial

from .base import *
from .brushes import Brush


class PaintTool(ImageToolBase):

    NAME = "paint"
    PARAMS = [
        *Brush.PARAMS,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._undo_state = None

    def config_changed(self):
        self.brush = Brush(
            type=self.config["type"],
            size=self.config["size"],
            color=self.config["color"],
            alpha=self.config["alpha"],
        )

    def mouse_event(self, event: MouseEvent):
        if event.button == Qt.LeftButton:

            if event.type == MouseEvent.Press:
                if self.layer.active and self.layer.image:
                    self._undo_state = self.layer.image.copy()
                else:
                    self._undo_state = None

            if event.type in (MouseEvent.Press, MouseEvent.Drag):

                if self.layer.active:
                    with self.layer.image_painter() as painter:

                        self.brush.apply(self.limage, painter, event.pos)

            elif event.type == MouseEvent.Release:
                if self._undo_state and self.layer.image:
                    self.project.push_undo_action(
                        "Paint",
                        partial(self.layer.set_image, self._undo_state),
                        partial(self.layer.set_image, self.layer.image.copy())
                    )
