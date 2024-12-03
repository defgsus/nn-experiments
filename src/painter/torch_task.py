import os
from typing import Optional, Generator

import torch
import torch.nn as nn
import torchvision.transforms.functional as VF

from src.train.experiment import load_experiment_trainer
from src.train import Trainer
from src.util.image import set_image_channels
from .image import Image
# from .painter import PaintSequence


class TorchTask:

    def __init__(self):
        from .painter import Painter
        self.trainer: Optional[Trainer] = None
        self.model: Optional[nn.Module] = None
        self.model_name: Optional[str] = None
        self.painter = Painter(worker=None)
        #self.model23 = ColorizeModel()
        self.model23 = ShinyTubesModel()

    def info(self) -> dict:
        return {
            "pid": os.getgid(),
            #"model_loaded": bool(self.model_name),
            #"model_name": self.model_name,
        }

    def apply_model(self, image: Image):
        with torch.no_grad():
            return self.model23(image.tensor.unsqueeze(0)).squeeze(0).detach().cpu()

    def paint(self, image: Image, sequence: "PaintSequence"):
        self.painter._paint(image, sequence)
        return image

    #def load_model(self, filename: str, device: str = "auto"):
    #    self.trainer = load_experiment_trainer(filename, device=device)
    #    self.trainer.load_checkpoint()
    #    self.model = self.trainer.model
    #    self.model_name = filename
    #    return self.info()



class ExperimentModel:

    def __init__(self, filename):
        self.filename = filename
        self._trainer = None

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = load_experiment_trainer(self.filename)
            self._trainer.load_checkpoint()
        return self._trainer

    @property
    def model(self):
        return self.trainer.model

    @property
    def device(self) -> torch.device:
        return self.trainer.device


class ColorizeModel(ExperimentModel):

    def __init__(self):
        super().__init__("experiments/img2img/colorize/colorize-ds.yml")

    def __call__(self, image: torch.Tensor):
        orig_channels = image.shape[-3]

        input = VF.rgb_to_grayscale(set_image_channels(image, 3)).to(self.device)
        self.model.eval()
        output = self.model(input).clamp(0, 1)

        return set_image_channels(output, orig_channels)


class ShinyTubesModel(ExperimentModel):

    def __init__(self):
        super().__init__("experiments/img2img/shinytubes.yml")

    def __call__(self, image: torch.Tensor):
        self.model.eval()
        return self.model(image.to(self.device)).clamp(0, 1)
