from .base import *


class PaintTool(ImageToolBase):

    NAME = "paint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mouse_event(self, event: MouseEvent):
        if event.type == MouseEvent.Drag:

            with self.layer.image_painter() as painter:

                rect = QRect(event.x - 2, event.y - 2, 4, 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(200, 50, 50, 100))
                painter.drawRect(rect)

