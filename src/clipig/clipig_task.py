from typing import Tuple, Optional, Union, Iterable, Generator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from ..models.clip import ClipSingleton
from ..util import to_torch_device
from .source_models import PixelModel
from .task_parameters import get_complete_task_config
from . import transformations


class ClipigTask:

    def __init__(
            self,
            config: dict,
    ):
        self.config = get_complete_task_config(config)
        self._original_config = config

        self._source_model: Optional[nn.Module] = None

    @property
    def clip_model_name(self) -> str:
        return self.config["clip_model_name"]

    @property
    def device_name(self) -> str:
        return self.config["device"]

    @property
    def device(self) -> torch.device:
        return to_torch_device(self.device_name)

    @property
    def num_iterations(self) -> int:
        return self.config["num_iterations"]

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

    def create_targets(self) -> List[dict]:
        targets = []
        for target_conf in self.config["targets"]:

            texts = [
                target_conf["prompt"]
            ]

            target_embeddings = self.clip_encode_text(texts)
            target_dots = torch.ones(target_conf["batch_size"], len(texts)).half().to(self.device)

            transforms = []
            for trans_conf in target_conf["transformations"]:
                klass = transformations.transformations[trans_conf["name"]]
                try:
                    trans = klass(**trans_conf["params"])
                except TypeError as e:
                    e.args = (*e.args, f"for class {klass}")
                    raise
                transforms.append(trans)

            targets.append({
                **target_conf,
                "transformations": VT.Compose(transforms),
                "target_embeddings": target_embeddings,
                "target_dots": target_dots,
            })

        return targets

    def run(self) -> Generator[dict, None, None]:
        yield {"status": "initializing"}

        source_model, optimizer = self.create_source_model()
        ClipSingleton.get(self.clip_model_name, self.device_name)

        yield {"status": "running"}

        targets = self.create_targets()

        for it in range(self.num_iterations):
            pixels = source_model().float()

            loss_sum = None
            for target in targets:
                transforms = target["transformations"]

                pixel_batch = []
                for i in range(target["batch_size"]):
                    pixel_batch.append(self._to_clip_size(transforms(pixels)).unsqueeze(0))
                pixel_batch = torch.concat(pixel_batch).half()

                image_embeddings = self.clip_encode_image(pixel_batch, requires_grad=True)
                # print(pixels.dtype, image_embeddings.dtype, target_embeddings.dtype)

                dots = image_embeddings @ target["target_embeddings"].T

                loss = F.l1_loss(dots, target["target_dots"])

                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            yield {
                "iteration": int,
                "loss": float(loss_sum),
                "pixels": pixels,
            }

    def _to_clip_size(self, pixels: torch.Tensor):
        if pixels.shape[-2:] != (224, 224):
            pixels = VF.resize(pixels, (224, 224), VT.InterpolationMode.NEAREST, antialias=False)
        return pixels
