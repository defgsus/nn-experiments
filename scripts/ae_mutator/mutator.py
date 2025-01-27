import copy
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.experiment import load_experiment_trainer


class Mutator:

    def __init__(
            self,
            experiment_file: str,
            device: str,
            width: int = 10,
            height: int = 10
    ):
        self.width, self.height = width, height
        self._trainer = load_experiment_trainer(experiment_file, device)
        self.model.eval()

        self.images = self.get_dataset_samples(width * height)
        self.latents = self.get_latents(self.images)
        self.generate_images()
        self._stack = []

        self.image_ids = [
            [y * width + x for x in range(width)]
            for y in range(height)
        ]
        self._id_counter = width * height

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def device(self) -> torch.device:
        return self._trainer.device

    def get_dataset_samples(self, count: int) -> torch.Tensor:
        loader = DataLoader(self._trainer.data_loader.dataset, batch_size=count)
        images = next(iter(loader))
        if isinstance(images, (list, tuple)):
            images = images[0]
        return images

    @torch.no_grad()
    def get_latents(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(images.to(self.device))

    def push(self):
        self._stack.append({
            "images": self.images.detach().clone().cpu(),
            "latents": self.latents.detach().clone().cpu(),
            "image_ids": copy.deepcopy(self.image_ids),
        })

    def pop(self):
        if self._stack:
            entry = self._stack.pop()
            self.images = entry["images"].to(self.device)
            self.latents = entry["latents"].to(self.device)
            self.image_ids = entry["image_ids"]

    def mutate(self, x: int, y: int, amount: float = 1.):
        idx = x + y * self.width
        source_latent = self.latents[idx]

        max_dist = math.sqrt(self.width ** 2 + self.height ** 2)
        for y_ in range(self.height):
            for x_ in range(self.width):
                idx_ = x_ + y_ * self.width
                dist = math.sqrt((x - x_) ** 2 + (y - y_) ** 2) / max_dist

                if idx != idx_:
                    self.latents[idx_] = self.mutate_latent(source_latent, dist * amount)
                    self.image_ids[y_][x_] = self._id_counter
                    self._id_counter += 1

        self.generate_images()

    def generate_images(self):
        self.images = self.model.decoder(self.latents).clamp(0, 1)

    @torch.no_grad()
    def mutate_latent(self, latent: torch.Tensor, amount: float) -> torch.Tensor:
        latent = latent.clone()

        prob = (torch.ones_like(latent) * amount).bernoulli() * amount

        latent = latent + torch.randn_like(latent) * prob

        return latent
