from typing import Tuple, Optional, Union, Iterable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from ..models.clip import ClipSingleton
from ..util import to_torch_device
from .source_models import PixelModel


class ClipigTask:

    def __init__(
            self,
            config: dict,
    ):
        self.config = config

        self._source_model: Optional[nn.Module] = None

    @property
    def clip_model_name(self) -> str:
        return self.config.get("clip_model_name") or "ViT-B/32"

    @property
    def device_name(self) -> str:
        return self.config.get("device") or "auto"

    @property
    def device(self) -> torch.device:
        return to_torch_device(self.device_name)

    @property
    def batch_size(self) -> int:
        return self.config.get("batch_size") or 1

    @property
    def num_iterations(self) -> int:
        return self.config.get("num_iterations") or 10

    def clip_encode_text(
            self,
            text: Union[str, Iterable[str]],
            truncate: bool = False,
            requires_grad: bool = False,
            normalize: bool = True,
    ):
        return ClipSingleton.encode_text(
            text=text,
            truncate=truncate,
            model=self.clip_model_name,
            device=self.device_name,
            requires_grad=requires_grad,
            normalize=normalize,
        )

    def clip_encode_image(
            self,
            image_batch: torch.Tensor,
            requires_grad: bool = False,
            normalize: bool = True,
    ):
        return ClipSingleton.encode_image(
            image_batch=image_batch,
            model=self.clip_model_name,
            device=self.device_name,
            requires_grad=requires_grad,
            normalize=normalize,
        )

    def create_source_model(self) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model = PixelModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)

        return model, optimizer

    def create_transforms(self):
        return VT.Compose([
            #VT.RandomErasing(),
            #VT.Pad(30, padding_mode="reflect"),
            #Debug("model"),
            #RandomWangMap((8, 8), probability=1, overlap=0, num_colors=2),
            #Debug("wangmap"),
            VT.RandomAffine(
                degrees=35.,
                #scale=(1., 3.),
                #scale=(.3, 1.),
                #translate=(0, 4. / 64.),
            ),
            VT.RandomCrop((224, 224)),
        ])

    def create_target_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        target_embeddings = self.clip_encode_text("top-down RPG style pixelart game")

        target_dots = torch.ones(self.batch_size, 1).half().to(self.device)

        return target_embeddings, target_dots

    def run(self) -> Generator[dict, None, None]:
        source_model, optimizer = self.create_source_model()

        transforms = self.create_transforms()

        target_embeddings, target_dots = self.create_target_embeddings()

        for it in range(self.num_iterations):
            pixel_batch = []
            pixels = source_model().float()

            for i in range(self.batch_size):
                pixel_batch.append(self._to_clip_size(transforms(pixels)).unsqueeze(0))
            pixel_batch = torch.concat(pixel_batch).half()

            image_embeddings = self.clip_encode_image(pixel_batch, requires_grad=True)
            # print(pixels.dtype, image_embeddings.dtype, target_embeddings.dtype)

            dots = image_embeddings @ target_embeddings.T

            loss = F.l1_loss(dots, target_dots)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yield {
                "iteration": int,
                "loss": float(loss),
                "pixels": pixels,
            }

    def _to_clip_size(self, pixels: torch.Tensor):
        if pixels.shape[-2:] != (224, 224):
            pixels = VF.resize(pixels, (224, 224), VT.InterpolationMode.NEAREST, antialias=False)
        return pixels
