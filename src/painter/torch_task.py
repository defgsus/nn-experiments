import os
from typing import Optional, Generator

import torch
import torch.nn as nn

from src.train.experiment import load_experiment_trainer
from src.train import Trainer
from .image import Image
# from .painter import PaintSequence


class TorchTask:

    def __init__(self):
        from .painter import Painter
        self.trainer: Optional[Trainer] = None
        self.model: Optional[nn.Module] = None
        self.model_name: Optional[str] = None
        self.painter = Painter(worker=None)
        self.model23 = nn.Conv2d(3, 3, 5, 1, 2)

    def info(self) -> dict:
        return {
            "pid": os.getgid(),
            #"model_loaded": bool(self.model_name),
            #"model_name": self.model_name,
        }

    def apply_model(self, image: Image):
        with torch.no_grad():
            return self.model23(image.tensor).detach()

    def paint(self, image: Image, sequence: "PaintSequence"):
        self.painter._paint(image, sequence)
        return image

    #def load_model(self, filename: str, device: str = "auto"):
    #    self.trainer = load_experiment_trainer(filename, device=device)
    #    self.trainer.load_checkpoint()
    #    self.model = self.trainer.model
    #    self.model_name = filename
    #    return self.info()

