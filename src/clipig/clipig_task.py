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
from .app.images.tiling import LImageTiling
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
            model.randomize(
                mean=self.config["init_random_mean"],
                std=self.config["init_random_var"],
            )

        elif init_method in ("input", "layer") and ((image := self.get_input_image()) is not None):
            model.set_image(image.to(self.device))

        else:
            model.clear()

        return model

    def create_targets(self, source_model: nn.Module) -> List[dict]:
        targets = []
        for target_conf in self.config["targets"]:
            if not target_conf["active"]:
                continue

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

                    elif feature.get("type") == "layer":
                        if feature.get("layer") is not None:
                            img = self.get_layer(feature["layer"])
                            if img is None:
                                print(f"Target feature layer '{feature['layer']}' not found")
                            else:
                                image_targets[len(target_embeddings)] = self._to_clip_pixels(img)
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
                    .to(self.device)
                )
                target_weights = (
                    torch.Tensor([[prompt["weight"] for prompt in target_features]])
                    .repeat(batch_size, 1)
                    .to(self.device)
                )

            else:
                target_embeddings = None
                target_dots = None
                target_weights = None

            transforms = []
            for trans_conf in target_conf["transformations"]:
                if trans_conf["params"]["active"]:
                    trans = transformations.create_transformation(trans_conf["name"], trans_conf["params"])
                    trans._clipig = self
                    transforms.append(trans)

            if self.task_type in (self.TaskType.T_CLIPIG, ):
                optimizer = create_optimizer(source_model.parameters(), target_conf["optimizer"])
            else:
                optimizer = None

            mask_layer = self.get_layer(target_conf["mask_layer"])
            if target_conf["mask_layer"]:
                if mask_layer is None:
                    print(f"Mask layer '{target_conf['mask_layer']}' not found")
                else:
                    mask_layer = mask_layer[:3].mean(dim=0, keepdim=True).clamp(0, 1)

            targets.append({
                **target_conf,
                "optimizer": optimizer,
                "learnrate": target_conf["optimizer"]["learnrate"],
                "transformations": VT.Compose(transforms) if transforms else lambda x: x,
                "target_embeddings": target_embeddings,
                "target_dots": target_dots,
                "target_weights": target_weights,
                "mask_layer": mask_layer,
            })

        return targets

    def run(self) -> Generator[dict, None, None]:

        yield {"status": "initializing"}

        self.source_model = source_model = self.create_source_model()
        if self.task_type == self.TaskType.T_CLIPIG:
            ClipSingleton.get(self.clip_model_name, self.device_name)

        targets = self.create_targets(source_model)
        gradient_layer = None

        yield {"status": "running"}

        if self.task_type == self.TaskType.T_TRANSFORMATION_PREVIEW:
            yield from self._preview_transformations(source_model, targets)
            return

        self._last_pixel_yield_time = 0
        for it in range(self.num_iterations):

            if self.config.get("dummy_mode"):
                yield {"pixels": source_model().detach()}

            loss_per_target = []
            for target_idx, target in enumerate(targets):
                pixels = source_model()
                pixels_backup = pixels.clone()

                optimizer: torch.optim.Optimizer = target["optimizer"]
                optimizer.zero_grad()

                for accum_idx in range(target["grad_accum_steps"]):
                    transforms = target["transformations"]

                    pixel_batch = []
                    for i in range(target["batch_size"]):
                        pixel_batch.append(self._to_clip_pixels(transforms(pixels)).unsqueeze(0))
                    pixel_batch = torch.concat(pixel_batch)

                    image_embeddings = self.clip_encode_image(pixel_batch, requires_grad=True)

                    dots = image_embeddings @ target["target_embeddings"].T

                    abs_error = (dots - target["target_dots"]).abs()
                    loss = (abs_error * target["target_weights"]).mean()

                    loss.backward()
                    loss_per_target.append(float(loss))

                self._clip_gradient(source_model, target)

                if self.config["output_gradient_layer"]:
                    with torch.no_grad():
                        params = list(source_model.parameters())
                        grad = params[0].grad.abs()
                        if grad.ndim == 3 and grad.shape[0] == 3:
                            grad = grad / target["grad_accum_steps"] / target["learnrate"]
                            if target["mask_layer"] is not None:
                                mask_layer = self._get_mask_layer(target, pixels_backup)
                                grad = grad * mask_layer
                            gradient_layer = self._accumulate_gradient_layer(gradient_layer, grad)

                optimizer.step()

                if target["mask_layer"] is not None:
                    mask_layer = self._get_mask_layer(target, pixels_backup)
                    with torch.no_grad():
                        pixels = source_model()
                        source_model.set_image(
                            pixels * mask_layer + (1 - mask_layer) * pixels_backup
                        )

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

                    if self.config["output_gradient_layer"] and gradient_layer is not None:
                        layer = gradient_layer
                        max_value = gradient_layer.max()
                        if max_value > 0:
                            layer = layer / max_value
                        message["gradient_layer"] = layer

            yield message

    def _accumulate_gradient_layer(self, accumulator: torch.Tensor, grad: torch.Tensor):
        if accumulator is None:
            return grad

        if self.config["gradient_layer_accumulation"] == "average":
            return accumulator + grad
        elif self.config["gradient_layer_accumulation"] == "moving_average":
            return accumulator + self.config["gradient_layer_smoothness"] * (grad - accumulator)
        elif self.config["gradient_layer_accumulation"] == "max":
            return torch.maximum(accumulator, grad)
        else:
            raise ValueError(f"Invalid gradient_layer_accumulation '{self.config['gradient_layer_accumulation']}'")

    def _clip_gradient(self, source_model: nn.Module, target: dict):
        if target["clip_grad_norm"]:
            torch.nn.utils.clip_grad_norm_(source_model.parameters(), target["clip_grad_norm"])

        if target["clip_grad_above_percent"] > 0 or target["clip_grad_below_percent"] > 0:
            for param in source_model.parameters():
                min_value = param.grad.abs().min()
                max_value = param.grad.abs().max()
                if target["clip_grad_below_percent"] > 0:
                    too_low_value = min_value * target["clip_grad_below_percent"] / 100
                    param.grad[param.grad < too_low_value] = 0.
                if target["clip_grad_above_percent"] > 0:
                    too_high_value = max_value * target["clip_grad_above_percent"] / 100
                    param.grad[param.grad > too_high_value] = 0.

    def _get_mask_layer(self, target: dict, pixels: torch.Tensor) -> torch.Tensor:
        if target["mask_layer"] is None:
            return pixels

        if target["mask_layer"].shape[1:] != pixels.shape:
            target["mask_layer"] = VF.resize(target["mask_layer"], pixels.shape[1:], VF.InterpolationMode.BICUBIC)
        return target["mask_layer"]

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
        model = ClipSingleton.get(
            model=self.clip_model_name,
            device=self.device_name,
        )

        if tuple(pixels.shape[-2:]) != model.shape[-2:]:
            pixels = image_resize_crop(pixels, model.shape[-2:])
        #    pixels = VF.resize(pixels, model.shape[-2], VT.InterpolationMode.NEAREST, antialias=False)

        if pixels.shape[0] != model.shape[0]:
            pixels = set_image_channels(pixels, model.shape[0])

        if pixels.dtype != model.dtype:
            pixels = set_image_dtype(pixels, model.dtype)

        return pixels.clamp(0, 1).to(self.device)

    def get_input_image(self) -> Optional[torch.Tensor]:
        if self.config["initialize"] == "input":
            image = self.config.get("input_image")
        elif self.config["initialize"] == "layer":
            image = self.get_layer(self.config["input_layer"])
        else:
            return None

        if image is None:
            return None

        if isinstance(image, torch.Tensor):
            return image

    def get_layer(self, name: str) -> Optional[torch.Tensor]:
        if not name:
            return
        layers = self.config.get("layers")
        if layers is None or name not in layers:
            return
        image = layers[name]
        if isinstance(image, torch.Tensor):
            return image.to(self.device)

    def input_tiling(self) -> Optional[LImageTiling]:
        return self.config.get("input_tiling")
