import time
from copy import deepcopy
from typing import Tuple, Optional, Union, Iterable, Generator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from ..models.clip import ClipSingleton
from ..util import to_torch_device
from .source_models import create_source_model
from .parameters import get_complete_clipig_task_config
from . import transformations
from src.util.image import set_image_channels, set_image_dtype


class ClipigTask:

    def __init__(
            self,
            config: dict,
    ):
        self.config = get_complete_clipig_task_config(config)
        self._original_config = config
        # minimum delay in seconds between yields of pixel data
        self._pixel_yield_delay_sec = 1.
        self._last_pixel_yield_time = 0

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

    def create_source_model(self) -> nn.Module:
        model = create_source_model(self.config["source_model"], device=self.device)

        init_method = self.config["initialize"]
        if init_method == "random":
            model.randomize()

        elif init_method == "input" and ((image := self.get_input_image()) is not None):
            model.set_image(image.to(self.device))

        else:
            model.clear()

        return model

    def create_targets(self, source_model: nn.Module) -> List[dict]:
        targets = []
        for target_conf in self.config["targets"]:

            texts = [
                target_conf["prompt"]
            ]

            target_embeddings = self.clip_encode_text(texts)
            target_dots = torch.ones(target_conf["batch_size"], len(texts)).half().to(self.device)

            transforms = []
            for trans_conf in target_conf["transformations"]:
                if trans_conf["params"]["active"]:
                    klass = transformations.transformations[trans_conf["name"]]

                    # remove extra parameters
                    trans_params = deepcopy(trans_conf["params"])
                    trans_params.pop("active")

                    try:
                        trans = klass(**trans_params)
                    except TypeError as e:
                        e.args = (*e.args, f"for class {klass}")
                        raise
                    transforms.append(trans)

            optimizer = torch.optim.Adam(source_model.parameters(), lr=target_conf["learnrate"])

            targets.append({
                **target_conf,
                "optimizer": optimizer,
                "transformations": VT.Compose(transforms) if transforms else lambda x: x,
                "target_embeddings": target_embeddings,
                "target_dots": target_dots,
            })

        return targets

    def run(self) -> Generator[dict, None, None]:
        yield {"status": "initializing"}

        source_model = self.create_source_model()
        ClipSingleton.get(self.clip_model_name, self.device_name)

        yield {"status": "running"}

        targets = self.create_targets(source_model)

        self._last_pixel_yield_time = 0
        for it in range(self.num_iterations):

            loss_per_target = []
            for target in targets:
                pixels = source_model().float()
                transforms = target["transformations"]

                pixel_batch = []
                for i in range(target["batch_size"]):
                    pixel_batch.append(self._to_clip_pixels(transforms(pixels)).unsqueeze(0))
                pixel_batch = torch.concat(pixel_batch).half()

                image_embeddings = self.clip_encode_image(pixel_batch, requires_grad=True)

                dots = image_embeddings @ target["target_embeddings"].T

                loss = F.l1_loss(dots, target["target_dots"])

                optimizer = target["optimizer"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_target.append(float(loss))

            message = {
                "iteration": int,
                "loss": sum(loss_per_target),
                "loss_per_target": loss_per_target,
            }
            cur_time = time.time()
            if cur_time - self._last_pixel_yield_time > self._pixel_yield_delay_sec:
                self._last_pixel_yield_time = cur_time
                with torch.no_grad():
                    message["pixels"] = source_model().detach()

            yield message

    def _to_clip_pixels(self, pixels: torch.Tensor):
        if pixels.shape[-2:] != (224, 224):
            pixels = VF.resize(pixels, (224, 224), VT.InterpolationMode.NEAREST, antialias=False)

        if pixels.shape[0] != 3:
            pixels = set_image_channels(pixels, 3)

        if pixels.dtype != torch.float16:
            pixels = set_image_dtype(pixels, torch.float16)

        return pixels

    def get_input_image(self) -> Optional[torch.Tensor]:
        image = self.config.get("input_image")
        if image is None:
            return

        if isinstance(image, torch.Tensor):
            return image