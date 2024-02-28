import time
from pathlib import Path
from copy import deepcopy
from typing import Tuple, Optional, Union, Iterable, Generator, List

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from ..models.clip import ClipSingleton
from ..util import to_torch_device
from .source_models import create_source_model, SourceModelBase
from .parameters import get_complete_clipig_task_config
from . import transformations
from .optimizers import create_optimizer
from src.util.image import set_image_channels, set_image_dtype, image_resize_crop


class ClipigTask:

    class TaskType:
        T_CLIPIG = "clipig"
        T_TRANSFORMATION_PREVIEW = "transformation_preview"

    def __init__(
            self,
            config: dict,
    ):
        self.config = get_complete_clipig_task_config(config)
        self._original_config = config
        # minimum delay in seconds between yields of pixel data
        self._pixel_yield_delay_sec = config.get("pixel_yield_delay_sec", 1.)
        self._last_pixel_yield_time = 0

        self.source_model: Optional[nn.Module] = None

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

    @property
    def task_type(self):
        return self.config.get("task_type") or self.TaskType.T_CLIPIG

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

    def load_image(self, filename: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        if isinstance(filename, torch.Tensor):
            return filename
        return VF.to_tensor(PIL.Image.open(str(filename)))

    def create_source_model(self) -> SourceModelBase:
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
            batch_size = target_conf["batch_size"]

            if self.task_type in (self.TaskType.T_CLIPIG, ):

                target_features = target_conf["target_features"]

                text_targets = {}
                image_targets = {}
                target_embeddings = []
                for i, feature in enumerate(target_features):
                    if feature.get("type") == "image":
                        if feature.get("image") is not None:
                            image_targets[len(target_embeddings)] = self._to_clip_pixels(
                                self.load_image(feature["image"])
                            )
                            target_embeddings.append(None)
                    elif feature.get("type") == "text":
                        if feature.get("text") is not None:
                            text_targets[len(target_embeddings)] = feature["text"]
                            target_embeddings.append(None)

                if image_targets:
                    for idx, emb in zip(
                            image_targets.keys(),
                            self.clip_encode_image(torch.concat([
                                img.unsqueeze(0) for img in image_targets.values()
                            ], dim=0))
                    ):
                        target_embeddings[idx] = emb
                if text_targets:
                    for idx, emb in zip(
                            text_targets.keys(),
                            self.clip_encode_text(list(text_targets.values()))
                    ):
                        target_embeddings[idx] = emb

                if not target_embeddings:
                    raise ValueError(f"no target_features defined")

                target_embeddings = torch.concat([e.unsqueeze(0) for e in target_embeddings])

                target_dots = (
                    torch.ones(batch_size, len(target_features))
                    .half().to(self.device)
                )
                target_weights = (
                    torch.Tensor([[prompt["weight"] for prompt in target_features]])
                    .repeat(batch_size, 1)
                    .half().to(self.device)
                )

            else:
                target_embeddings = None
                target_dots = None
                target_weights = None

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

            if self.task_type in (self.TaskType.T_CLIPIG, ):
                optimizer = create_optimizer(source_model.parameters(), target_conf["optimizer"])
            else:
                optimizer = None

            targets.append({
                **target_conf,
                "optimizer": optimizer,
                "transformations": VT.Compose(transforms) if transforms else lambda x: x,
                "target_embeddings": target_embeddings,
                "target_dots": target_dots,
                "target_weights": target_weights,
            })

        return targets

    def run(self) -> Generator[dict, None, None]:

        yield {"status": "initializing"}

        self.source_model = source_model = self.create_source_model()
        if self.task_type == self.TaskType.T_CLIPIG:
            ClipSingleton.get(self.clip_model_name, self.device_name)

        targets = self.create_targets(source_model)

        yield {"status": "running"}

        if self.task_type == self.TaskType.T_TRANSFORMATION_PREVIEW:
            yield from self._preview_transformations(source_model, targets)
            return

        self._last_pixel_yield_time = 0
        for it in range(self.num_iterations):

            if self.config.get("dummy_mode"):
                yield {"pixels": source_model().detach()}

            loss_per_target = []
            for target in targets:
                pixels = source_model()
                transforms = target["transformations"]

                pixel_batch = []
                for i in range(target["batch_size"]):
                    pixel_batch.append(self._to_clip_pixels(transforms(pixels)).unsqueeze(0))
                pixel_batch = torch.concat(pixel_batch).half()

                image_embeddings = self.clip_encode_image(pixel_batch, requires_grad=True)

                dots = image_embeddings @ target["target_embeddings"].T

                abs_error = (dots - target["target_dots"]).abs()
                loss = (abs_error * target["target_weights"]).mean()

                # loss = F.l1_loss(dots, target["target_dots"])

                optimizer: torch.optim.Optimizer = target["optimizer"]
                optimizer.zero_grad()
                loss.backward()
                if target["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(source_model.parameters(), target["clip_grad_norm"])
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

    @torch.no_grad()
    def _preview_transformations(
            self,
            source_model: SourceModelBase,
            targets: List[dict],
            batch_size: int = 5,
    ) -> Generator[dict, None, None]:
        pixels = source_model()

        for target_idx, target in enumerate(targets):
            transforms = target["transformations"]

            pixel_batch = []
            for i in range(batch_size):
                pixel_batch.append(self._to_clip_pixels(transforms(pixels)))

            yield {"transformation_preview": {"target_index": target_idx, "pixels": pixel_batch}}

    def _to_clip_pixels(self, pixels: torch.Tensor):
        if tuple(pixels.shape[-2:]) != (224, 224):
            pixels = image_resize_crop(pixels, (224, 224))
        #    pixels = VF.resize(pixels, (224, 224), VT.InterpolationMode.NEAREST, antialias=False)

        if pixels.shape[0] != 3:
            pixels = set_image_channels(pixels, 3)

        if pixels.dtype != torch.float16:
            pixels = set_image_dtype(pixels, torch.float16)

        return pixels.clamp(0, 1)

    def get_input_image(self) -> Optional[torch.Tensor]:
        image = self.config.get("input_image")
        if image is None:
            return

        if isinstance(image, torch.Tensor):
            return image
