from functools import partial

from .base import *
from ..tiling import LImageTiling


class SelectionTool(ImageToolBase):

    NAME = "select"


class MoveTool(ImageToolBase):

    NAME = "move"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_pos = None
        self._mouse_pos = None
        
    def mouse_event(self, event: MouseEvent):
        if event.type == MouseEvent.Press:
            self._layer_pos = self.layer.position
            self._mouse_pos = event.pos

        if event.type == MouseEvent.Drag and self._layer_pos:
            self.layer.set_position((
                self._layer_pos[0] + event.x - self._mouse_pos[0],
                self._layer_pos[1] + event.y - self._mouse_pos[1],
            ))

