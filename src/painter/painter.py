import math
from typing import Tuple, Optional

import torch

from .image import Image
from .util import *
from .torch_worker import TorchWorker


class PaintSequence:

    def __init__(self):
        self._sequence = []

    def clear(self):
        self._sequence.clear()

    def add(self, x: int, y: int):
        self._sequence.append((x, y))

    def __iter__(self):
        yield from self._sequence

    def __bool__(self):
        return bool(self._sequence)

    def subtract(self, x: int, y: int) -> "PaintSequence":
        seq = PaintSequence()
        seq._sequence = [
            (ox - x, oy - y)
            for ox, oy in self._sequence
        ]
        return seq

    def bounding_box(self) -> Optional[BoundingBox]:
        if not self._sequence:
            return
        minx, miny, maxx, maxy = None, None, None, None
        for x, y in self:
            if minx is None or x < minx:
                minx = x
            if maxx is None or x > maxx:
                maxx = x
            if miny is None or y < miny:
                miny = y
            if maxy is None or y > maxy:
                maxy = y
        return BoundingBox(minx, miny, maxx + 1, maxy + 1)


class Painter:

    def __init__(self, worker: Optional[TorchWorker]):
        self._current_tool = NNTool()
        self._sequence = PaintSequence()
        self._worker = worker
        self._async_calls = dict()

    def paint(self, x: int, y: int):
        self._sequence.add(x, y)

    def update(self, image: Image) -> Optional[BoundingBox]:
        if self._worker is not None:
            for event in self._worker.poll():
                if event.get("call"):
                    event = event["call"]
                    async_call = self._async_calls.pop(event["uuid"])
                    if event.get("result"):
                        image_rect = event["result"]
                        paint_box = async_call["paint_box"]
                        async_call["image"] = image.tensor[:, paint_box.y1: paint_box.y2, paint_box.x1: paint_box.x2] = image_rect.tensor
                        return paint_box

        return_box = None
        if self._sequence:
            image_box = image.bounding_box
            paint_box = self.get_bounding_box(self._sequence)
            paint_box = paint_box.inside(image_box)
            if not paint_box.is_empty:
                image_rect = image.crop(paint_box)
                sequence = self._sequence.subtract(paint_box.x1, paint_box.y1)
                if self._current_tool.is_async and self._worker is not None:
                    id = self._worker.call("paint", image_rect, sequence)
                    self._async_calls[id] = {"image": image, "paint_box": paint_box}
                else:
                    self._paint(image_rect, sequence)
                    image.tensor[:, paint_box.y1: paint_box.y2, paint_box.x1: paint_box.x2] = image_rect.tensor
                    return_box = paint_box

            self._sequence.clear()
        return return_box

    def get_bounding_box(
            self,
            sequence: PaintSequence,
    ) -> Optional[BoundingBox]:
        return self._current_tool.get_bounding_box(sequence)

    def _paint(
            self,
            image: Image,
            sequence: PaintSequence,
    ):
        # print("PAINT", image.tensor.shape, sequence._sequence)

        self._current_tool.paint(image, sequence)
        return

        dx, dy = delta
        if dx == dy == 0:
            return tool.apply(self.image, pos, delta)

        full_box = None
        x, y = pos
        adx, ady = abs(dx), abs(dy)
        num = max(adx, ady)
        for i in range(num):
            box = tool.apply(self.image, (int(x), int(y)))
            if box:
                if full_box is None:
                    full_box = box
                else:
                    full_box = full_box.union(box)

            x += dx / num
            y += dy / num

        return full_box


class PaintToolBase:

    is_async = False

    def __init__(self):
        pass

    def get_bounding_box(
            self,
            sequence: PaintSequence,
    ) -> Optional[BoundingBox]:
        raise NotImplementedError

    def paint(
            self,
            image: Image,
            sequence: PaintSequence,
    ):
        raise NotImplementedError


class PixelBrushTool(PaintToolBase):

    def __init__(self):
        super().__init__()
        self.color = (.8, .9, 1.)

    def get_bounding_box(
            self,
            sequence: PaintSequence,
    ) -> Optional[BoundingBox]:
        return sequence.bounding_box()

    def paint(
            self,
            image: Image,
            sequence: PaintSequence,
    ):
        for x, y in sequence:
            if 0 <= y < image.height and 0 <= x < image.width:
                image.tensor[:, y, x] = torch.Tensor(self.color).to(image)


class BrushTool(PaintToolBase):

    def __init__(self, mask_size: int = 51):
        super().__init__()
        self.color = (.8, .9, 1.)
        self.alpha = .1

        size = self.mask_size = mask_size
        x = torch.meshgrid([torch.linspace(-1, 1, size), torch.linspace(-1, 1, size)], indexing="ij")[0]
        x = 1. - (x.pow(2.) * 2).sqrt() / math.sqrt(2) * (1. - 1. / size)
        self.mask = x * x.T

    def get_bounding_box(
            self,
            sequence: PaintSequence,
    ) -> Optional[BoundingBox]:
        box = sequence.bounding_box()
        if box is not None:
            h, w = self.mask.shape[-2:]
            hh = int(math.floor(h / 2))
            hw = int(math.floor(w / 2))
            return BoundingBox(box.x1 - hw, box.y1 - hw, box.x2 + hh, box.y2 + hh)

    def paint(
            self,
            image: Image,
            sequence: PaintSequence,
    ):
        for x, y in sequence:
            h, w = self.mask.shape[-2:]
            hh = int(math.floor(h / 2))
            hw = int(math.floor(w / 2))
            x1 = x - hw
            y1 = y - hh

            image_rect = image.bounding_box
            mask_rect = BoundingBox.from_tensor(self.mask)
            mask_rect = mask_rect.fit_into_at(image_rect, x1, y1)
            if mask_rect is not None:

                mask = mask_rect.crop_tensor(self.mask)

                self.apply_mask(image, mask, x1 + mask_rect.x1, y1 + mask_rect.y1)

    def apply_mask(self, image: Image, mask: torch.Tensor, x: int, y: int):
        x2 = x + mask.shape[-1]
        y2 = y + mask.shape[-2]
        color = torch.ones(3, *mask.shape) * torch.Tensor(self.color)[:, None, None]

        mask = mask * self.alpha
        mask = mask[None, :, :].repeat(3, 1, 1)

        image_rect = image.tensor[:, y:y2, x:x2]
        color = 1. - image_rect

        image.tensor[:, y:y2, x:x2] += mask * (color - image_rect)


class NNTool(BrushTool):

    is_async = True

    def __init__(self):
        super().__init__(32)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from src.train.experiment import load_experiment_trainer
            self._trainer = load_experiment_trainer(
                "experiments/diffusion/noisediffusion.yml",
                device="cpu",
            )
            self._trainer.load_checkpoint()
            self._model = self._trainer.model
        return self._model

    def apply_mask(self, image: Image, mask: torch.Tensor, x: int, y: int):
        from experiments.diffusion.trainer import DiffusionModelInput
        x2 = x + mask.shape[-1]
        y2 = y + mask.shape[-2]

        mask = mask * self.alpha
        mask = mask[None, :, :].repeat(3, 1, 1)

        image_rect = image.tensor[:, y:y2, x:x2]

        with torch.no_grad():
            noise = self.model(
                DiffusionModelInput(
                    image_rect.unsqueeze(0) * 2. - 1.,
                    torch.zeros(1, 1).to(image.tensor),
                    torch.zeros(1, 10).to(image.tensor),
                )
            ).noise
            color = self._trainer.diffusion_sampler.remove_noise(
                image_rect.unsqueeze(0) * 2. - 1., noise
            ).squeeze(0) * .5 + .5

        image.tensor[:, y:y2, x:x2] += mask * (color - image_rect)
