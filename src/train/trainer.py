import json
import os
import re
import math
import random
import itertools
import argparse
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union, Generator, Dict

from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util import to_torch_device, num_module_parameters
from src.util.image import signed_to_image, get_images_from_iterable
from src.models.util import get_loss_callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Trainer:

    def __init__(
            self,
            experiment_name: str,
            model: torch.nn.Module,
            data_loader: DataLoader,
            validation_loader: Optional[DataLoader] = None,
            freeze_validation_set: bool = False,
            min_loss: Optional[float] = None,
            max_epoch: Optional[int] = None,
            max_inputs: Optional[int] = None,
            optimizers: Iterable[torch.optim.Optimizer] = tuple(),
            schedulers: Iterable[torch.optim.lr_scheduler.LRScheduler] = tuple(),
            num_inputs_between_validations: Optional[int] = None,
            num_epochs_between_validations: Optional[int] = None,
            num_inputs_between_checkpoints: Optional[int] = None,
            num_epochs_between_checkpoints: int = 1,
            training_noise: float = 0.,
            train_input_transforms: Optional[Iterable[Callable]] = None,
            loss_function: Union[str, Callable, torch.nn.Module] = "l1",
            gradient_clipping: Optional[float] = None,
            gradient_accumulation: int = 1,
            num_train_loss_steps: int = 1000,
            reset: bool = False,
            device: Union[None, str, torch.DeviceObjType] = None,
            hparams: Optional[dict] = None,
            model_forward_kwargs: Optional[dict] = None,
            weight_image_kwargs: Optional[dict] = None,
            extra_description_values: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.model = model
        self.data_loader = data_loader
        self.validation_loader = validation_loader
        self.freeze_validation_set = freeze_validation_set
        self._validation_batches: Optional[torch.Tensor] = None
        self._best_validation_loss: Optional[float] = None
        self.min_loss = min_loss
        self.max_epoch = max_epoch
        self.max_inputs = max_inputs
        self.optimizers = list(optimizers)
        self.schedulers = list(schedulers)
        self.training_noise = training_noise
        self.loss_function = get_loss_callable(loss_function)
        self.num_train_loss_steps = num_train_loss_steps
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation = gradient_accumulation
        self.hparams = hparams
        self.model_forward_kwargs = model_forward_kwargs or {}
        self.weight_image_kwargs = weight_image_kwargs
        self.extra_description_values = extra_description_values
        self.num_inputs_between_validations = num_inputs_between_validations
        self.num_epochs_between_validations = num_epochs_between_validations
        self.num_inputs_between_checkpoints = num_inputs_between_checkpoints
        self.num_epochs_between_checkpoints = num_epochs_between_checkpoints
        self.epoch = 0
        self.num_batch_steps = 0
        self.num_input_steps = 0
        self._train_input_transforms = None if train_input_transforms is None else list(train_input_transforms)
        self.for_validation = False
        self._train_start_time = 0.
        self._train_auxiliary_time = 0.
        self._logged_scalars = {}

        self.tensorboard_path = PROJECT_ROOT / "runs" / self.experiment_name
        self.checkpoint_path = PROJECT_ROOT / "checkpoints" / self.experiment_name
        self.device = to_torch_device(device)
        self._skip_optimizer: Optional[List[bool]] = None
        self._loss_history = []
        self._loss_steps = 0
        self.last_validation_loss: Optional[float] = None

        try:
            self.model = self.model.to(self.device)
        except ValueError as e:
            if "8-bit" not in str(e):
                raise

        if reset:
            if self.tensorboard_path.exists():
                shutil.rmtree(self.tensorboard_path)
            if self.checkpoint_path.exists():
                shutil.rmtree(self.checkpoint_path)
        self.writer = SummaryWriter(str(self.tensorboard_path))

        self._every_callbacks = []
        self._setup_every_callbacks()

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Default implementation expects input_batch to be a tuple of two Tensors
        and returns the MSE loss between second tensor and model output.

        Override to implement something else and return a 0-dim loss tensor
        or a dict with multiple losses (for logging), e.g:
            {
                "loss": real_loss + additional_loss,
                "loss_real": real_loss,
                "additional_loss": additional_loss,
            }
        """
        input, target_features = input_batch[:2]
        if not self.for_validation:
            input = self.transform_input_batch(input)
        output_features = self.model(input, **self.model_forward_kwargs)

        return self.loss_function(output_features, target_features)

    def validation_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._train_step(input_batch)

    def write_step(self):
        pass

    def load_checkpoint(self, name: str = "snapshot") -> bool:
        checkpoint_filename = self.checkpoint_path / f"{name}.pt"

        if not checkpoint_filename.exists():
            return False

        print(f"loading {checkpoint_filename}")
        checkpoint_data = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint_data["state_dict"])
        if checkpoint_data.get("epoch"):
            self.epoch = checkpoint_data["epoch"] + 1
        else:
            self.epoch = 0
        self.num_batch_steps = checkpoint_data.get("num_batch_steps") or 0
        self.num_input_steps = checkpoint_data.get("num_input_steps") or 0

        if checkpoint_data.get("optimizers"):
            for opt, saved_state in zip(self.optimizers, checkpoint_data["optimizers"]):
                opt.load_state_dict(saved_state)

        if checkpoint_data.get("schedulers"):
            for sched, saved_state in zip(self.schedulers, checkpoint_data["schedulers"]):
                sched.load_state_dict(saved_state)

        best_filename = self.checkpoint_path / "best.json"
        if best_filename.exists():
            try:
                self._best_validation_loss = json.loads(best_filename.read_text())["validation_loss"]
                print(f"best validation loss so far: {self._best_validation_loss}")
            except (json.JSONDecodeError, KeyError):
                pass

        return True

    def save_checkpoint(self, name: str = "snapshot"):
        start_time_aux = time.time()
        checkpoint_filename = self.checkpoint_path / f"{name}.pt"
        print(f"storing {checkpoint_filename}")

        os.makedirs(checkpoint_filename.parent, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "epoch": self.epoch,
                "num_batch_steps": self.num_batch_steps,
                "num_input_steps": self.num_input_steps,
                "optimizers": [
                    opt.state_dict()
                    for opt in self.optimizers
                ],
                "schedulers": [
                    sched.state_dict()
                    for sched in self.schedulers
                ]
            },
            checkpoint_filename,
        )
        self._train_auxiliary_time += time.time() - start_time_aux

    def save_description(self, name: str = "description", extra: Optional[dict] = None):
        description_filename = self.checkpoint_path / f"{name}.json"
        print(f"storing {description_filename}")
        os.makedirs(description_filename.parent, exist_ok=True)
        description_filename.write_text(json.dumps({
            "experiment_name": self.experiment_name,
            "trainable_parameters": self.num_trainable_parameters(),
            "max_epoch": self.max_epoch,
            "max_inputs": self.max_inputs,
            "model": repr(self.model),
            "optimizers": [repr(o) for o in self.optimizers],
            "num_inputs": self.num_input_steps,
            "training_time": time.time() - self._train_start_time - self._train_auxiliary_time,
            "auxiliary_time": self._train_auxiliary_time,
            "scalars": self._logged_scalars,
            **(self.extra_description_values or {}),
            **(extra or {}),
        }, indent=2))

    def num_trainable_parameters(self) -> (int, int):
        trainable_count = 0
        for opt in self.optimizers:
            trainable_count += sum(
                sum(math.prod(p.shape) for p in g["params"] if p.requires_grad)
                for g in opt.param_groups
            )

        return (
            trainable_count,
            num_module_parameters(self.model)
        )

    def log_scalar(self, tag: str, value):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.num_input_steps)
        if isinstance(value, torch.Tensor):
            value = float(value.detach())
        self._logged_scalars[tag] = {"step": self.num_input_steps, "value": value}

    def log_image(self, tag: str, image: torch.Tensor):
        try:
            self.writer.add_image(tag=tag, img_tensor=image, global_step=self.num_input_steps)
        except TypeError as e:
            warnings.warn(f"logging image `{tag}` failed, {type(e).__name__}: {e}")

    def log_embedding(self, tag: str, embedding: torch.Tensor):
        self.writer.add_embedding(tag=tag, mat=embedding, global_step=self.num_input_steps)

    def train(self):
        print(f"---- training '{self.experiment_name}' on {self.device} ----")
        num_train_params, num_params = self.num_trainable_parameters()
        if num_train_params == num_params:
            print(f"trainable params: {num_train_params:,}")
        else:
            print(f"trainable params: {num_train_params:,} of {num_params:,}")

        self._train_start_time = time.time()
        self._train_auxiliary_time = 0.
        self._logged_scalars.clear()

        last_validation_step = None
        last_validation_epoch = None
        last_checkpoint_epoch = -self.num_epochs_between_checkpoints
        if self.num_inputs_between_validations is not None:
            last_validation_step = -self.num_inputs_between_validations
        if self.num_epochs_between_validations is not None:
            last_validation_epoch = -self.num_epochs_between_validations

        last_checkpoint_step = None
        if self.num_inputs_between_checkpoints is not None:
            last_checkpoint_step = 0

        self.model.train(True)

        _optimizer_step_called = False
        self.running = True
        has_run_validation = False
        while self.running:

            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                break

            total = None
            try:
                total = len(self.data_loader.dataset)
            except:
                pass

            last_optimizer_batch_idx = -1
            with tqdm(
                    total=total,
                    desc=f"epoch #{self.epoch}",
            ) as progress:
                has_run_validation = False
                for batch_idx, input_batch in enumerate(self.iter_training_batches()):
                    if not isinstance(input_batch, (tuple, list)):
                        input_batch = (input_batch, )

                    input_batch = [
                        i.to(self.device) if callable(getattr(i, "to", None)) else i
                        for i in input_batch
                    ]

                    if isinstance(input_batch[0], torch.Tensor):
                        input_batch_size = input_batch[0].shape[0]
                    elif isinstance(input_batch[0], dict):
                        v = input_batch[0][next(iter(input_batch[0].keys()))]
                        input_batch_size = len(v)
                    elif isinstance(input_batch[0], list):
                        input_batch_size = len(input_batch[0])
                    elif isinstance(input_batch[0], str):
                        input_batch_size = len(input_batch)
                    elif callable(getattr(input_batch[0], "__len__", None)):
                        input_batch_size = len(input_batch[0])
                    else:
                        raise TypeError(f"Can't determine input format from type '{type(input_batch[0]).__name__}'")

                    if self.training_noise > 0.:
                        with torch.no_grad():
                            input_batch[0] = input_batch[0] + self.training_noise * torch.randn_like(input_batch[0])

                    if self.epoch == 0 and batch_idx == 0:
                        if isinstance(input_batch, list) and isinstance(input_batch[0], str):
                            print(f" BATCH {len(input_batch)}, {len(input_batch[0])}")
                        else:
                            print(" BATCH", ", ".join(
                                str(b.shape) if hasattr(b, "shape") else "?"
                                for b in input_batch
                            ))

                    loss_result = self._train_step(input_batch)
                    if not isinstance(loss_result, dict):
                        loss_result = {"loss": loss_result}

                    progress.update(input_batch_size)

                    #(loss_result["loss"] / self.gradient_accumulation).backward()
                    loss_result["loss"].backward()

                    if (batch_idx + 1) % self.gradient_accumulation == 0:
                        last_optimizer_batch_idx = batch_idx
                        _optimizer_step_called = True
                        self._optimizer_step()

                    if _optimizer_step_called:
                        for i, sched in enumerate(self.schedulers):
                            if not self._skip_optimizer or not self._skip_optimizer[i]:
                                sched.step()

                                lr = sched.get_last_lr()
                                if isinstance(lr, (list, tuple)):
                                    lr = lr[0]
                                self._loss_history.append({f"learnrate_{i+1}_{type(sched.optimizer).__name__}": lr})

                    self.num_batch_steps += 1
                    self.num_input_steps += input_batch_size

                    self._loss_history.append({
                        key: float(value)
                        for key, value in loss_result.items()
                    })
                    self._loss_steps += input_batch_size
                    if self._loss_steps >= self.num_train_loss_steps:
                        losses = {}
                        for entry in self._loss_history:
                            for key, value in entry.items():
                                if key not in losses:
                                    losses[key] = []
                                losses[key].append(value)
                        for key, values in losses.items():
                            value = sum(values) / len(values)
                            self.log_scalar(f"train_{key}", value)
                            if key == "loss":
                                progress.desc=f"epoch #{self.epoch} (loss {value:.4f})"

                        self._loss_history.clear()
                        self._loss_steps = 0

                    if self.num_inputs_between_validations is not None:
                        if self.num_input_steps - last_validation_step >= self.num_inputs_between_validations:
                            last_validation_step = self.num_input_steps
                            last_validation_epoch = self.epoch
                            has_run_validation = True
                            self.run_validation()

                    if last_checkpoint_step is not None:
                        if self.num_input_steps - last_checkpoint_step >= self.num_inputs_between_checkpoints:
                            last_checkpoint_step = self.num_input_steps
                            self.save_checkpoint()

                    # print(f"\nepoch {self.epoch}, batch {self.num_batch_steps}, image {self.num_input_steps}, loss {float(loss)}")

                    self._run_every_callbacks()

                    if self.max_inputs and self.num_input_steps >= self.max_inputs:
                        print(f"max_inputs ({self.max_inputs}) reached")
                        self.running = False
                        break

            if last_optimizer_batch_idx != batch_idx:
                self._optimizer_step()

            if self.epoch - last_checkpoint_epoch >= self.num_epochs_between_checkpoints:
                last_checkpoint_epoch = self.epoch
                # self.save_checkpoint(f"epoch-{epoch:03d}")
                self.save_checkpoint()

            if self.num_epochs_between_validations is not None:
                if self.epoch - last_validation_epoch >= self.num_epochs_between_validations:
                    last_validation_step = self.num_input_steps
                    last_validation_epoch = self.epoch
                    has_run_validation = True
                    self.run_validation()

            if self.running:
                self.epoch += 1

        if not has_run_validation:
            self.run_validation()

        self.writer.flush()

    def _optimizer_step(self):
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.gradient_clipping,
                #error_if_nonfinite=True,
            )

        for i, opt in enumerate(self.optimizers):
            if not self._skip_optimizer or not self._skip_optimizer[i]:
                opt.step()

        for i, opt in enumerate(self.optimizers):
            if not self._skip_optimizer or not self._skip_optimizer[i]:
                opt.zero_grad()

    def iter_training_batches(self) -> Generator:
        """
        Override to adjust the training data or labels
        """
        for data in self.data_loader:
            yield self._to_device(data)

    def iter_validation_batches(self) -> Generator:
        """Makes a copy of the validation set to memory if 'freeze_validation_set' is True"""
        if self.validation_loader is None:
            return

        if not self.freeze_validation_set:
            for data in self.validation_loader:
                yield self._to_device(data)
            return

        if self._validation_batches is None:
            self._validation_batches = list(self.validation_loader)

        for data in self._validation_batches:
            yield self._to_device(data)

    def validation_sample(self, index: int) -> Optional[tuple]:
        if self.validation_loader is None:
            return

        if not self.freeze_validation_set:
            return self.validation_loader.dataset[index]

        if self._validation_batches is None:
            self._validation_batches = list(self.validation_loader)

        batch_size = self.validation_loader.batch_size
        batch_tuple = self._validation_batches[index // batch_size]
        b_index = index % batch_size
        return tuple(b[b_index] for b in batch_tuple)

    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return tuple(self._to_device(d) for d in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def transform_input_batch(self, input_batch):
        if self._train_input_transforms is not None:
            for transform in self._train_input_transforms:
                if isinstance(input_batch, (list, tuple)):
                    try:
                        input_batch = transform(input_batch)
                    except TypeError:
                        input_batch = transform(input_batch[0]), *input_batch[1:]
                else:
                    input_batch = transform(input_batch)

        return input_batch

    def run_validation(self):
        if self.validation_loader is None:
            return

        start_time_aux = time.time()

        print(f"validation @ {self.num_input_steps:,}")
        with torch.no_grad():
            self.for_validation = True
            self.model.for_validation = True
            self.model.eval()

            try:
                losses = {}
                for validation_batch in self.iter_validation_batches():
                    if not isinstance(validation_batch, (tuple, list)):
                        validation_batch = validation_batch,

                    validation_batch = [
                        i.to(self.device) if callable(getattr(i, "to", None)) else i
                        for i in validation_batch
                    ]

                    loss = self.validation_step(validation_batch)
                    if not isinstance(loss, dict):
                        loss = {"loss": loss}

                    for key, value in loss.items():
                        if key not in losses:
                            losses[key] = []
                        losses[key].append(value)

                losses = {
                    key: float(torch.Tensor(value).mean())
                    for key, value in losses.items()
                }

            finally:
                self.model.train(True)
                self.model.for_validation = False
                self.for_validation = False

            # print(f"\nVALIDATION loss {float(loss)}")
            for key, value in losses.items():
                self.log_scalar(f"validation_{key}", value)

            loss = self.last_validation_loss = losses["loss"]

            if self._best_validation_loss is None or loss < self._best_validation_loss:
                self._best_validation_loss = loss
                self.save_checkpoint("best")
                self.save_description("best", extra={"validation_loss": loss})

            self.save_checkpoint("snapshot")
            self.save_description("snapshot", extra={"validation_loss": loss})

            self.save_weight_image()
            self._write_step()

            if self.min_loss is not None and loss <= self.min_loss:
                print(f"min_loss ({self.min_loss}) reached with validation loss {loss}")
                self.running = False

        self._train_auxiliary_time += time.time() - start_time_aux

    def save_weight_image(
            self,
            max_single_shape: Iterable[int] = (512, 512),
            nrow: int = 16,
    ):
        if not hasattr(self.model, "weight_images"):
            self._save_default_weight_image()
            return

        def _make2d(vec):
            size = vec.shape[-1]
            center = int(math.sqrt(size))
            div = 1
            for i in range(size - center):
                f = size / (center + i)
                if f == int(f):
                    div = center + i
                    break
                f = size / (center - i)
                if f == int(f):
                    div = center - i
                    break
            return vec.view(div, size // div)

        images = self.model.weight_images(**(self.weight_image_kwargs or {}))
        if images is None:
            return

        max_shape = None
        max_single_shape = list(max_single_shape)
        for image_idx, image in enumerate(images):

            if image.ndim == 1:
                images[image_idx] = image = _make2d(image)

            if any(a > b for a, b in zip(image.shape, max_single_shape)):
                image = VF.crop(image, 0, 0, min(image.shape[-2], max_single_shape[-2]), min(image.shape[-1], max_single_shape[-1]))
                images[image_idx] = image

            if max_shape is None:
                max_shape = list(image.shape)
            else:
                for i in range(len(max_shape)):
                    max_shape[i] = max(max_shape[i], image.shape[i])

        for image_idx, image in enumerate(images):

            if any(a < b for a, b in zip(image.shape, max_shape)):
                images[image_idx] = VF.pad(image, [0, 0, max_shape[-1] - image.shape[-1], max_shape[0] - image.shape[0]])

        if not len(images):
            self._save_default_weight_image()
            return

        norm_grid = make_grid(
            [i.unsqueeze(0) if i.ndim == 2 else i for i in images],
            nrow=nrow, normalize=True
        )
        signed_grid = make_grid(
            [signed_to_image(i) for i in images],
            nrow=nrow,
        )

        self.log_image("weights", make_grid([signed_grid, norm_grid]))

    def _save_default_weight_image(self):
        from src.models.util import get_model_weight_images
        kwargs = self.weight_image_kwargs or {}
        grid1 = get_model_weight_images(self.model, **{**kwargs, "normalize": "each"})
        grid2 = get_model_weight_images(self.model, **{**kwargs, "normalize": "all"})
        self.log_image("weights", make_grid([grid1, grid2], nrow=1))

    @classmethod
    def add_parser_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "experiment_name", type=str,
            help="name of the experiment for checkpoints and tensorboard logs"
        )
        parser.add_argument(
            "-r", "--reset", type=bool, nargs="?", default=False, const="True",
            help="Delete previous checkpoint and logs"
        )
        parser.add_argument(
            "-d", "--device", type=str, nargs="?", default="auto",
            help="Specify device"
        )

    def _train_step(self, input_batch: Tuple[torch.Tensor, ...]) -> Union[torch.Tensor, dict]:
        #torch.set_grad_enabled(True)
        self.model.train(True)

        if callable(getattr(self.model, "before_train_step", None)):
            self.model.before_train_step(self)

        if hasattr(self.model, "train_step"):
            return self.model.train_step(input_batch)
        else:
            return self.train_step(input_batch)

    def _write_step(self):
        start_time_aux = time.time()
        if hasattr(self.model, "write_step"):
            self.model.write_step(self)
        else:
            self.write_step()
        self._train_auxiliary_time += time.time() - start_time_aux

    def _setup_every_callbacks(self):
        self._every_callbacks = []
        for name in dir(self):
            if name.startswith("every_") and callable(getattr(self, name)):
                parts = name.split("_")

                num = int(parts[1].lstrip("0") or 0)
                if len(parts) > 2:
                    what = parts[2]
                else:
                    what = "inputs"

                if what == "epoch":
                    what = "epochs"

                assert what in ("inputs", "batches", "epochs"), f"Got '{what}'"

                self._every_callbacks.append({
                    "what": what,
                    "num": num,
                    "last_num": 0,
                    "callable": getattr(self, name),
                })

    def _run_every_callbacks(self):
        start_time_aux = time.time()
        for every in self._every_callbacks:
            if every["what"] == "inputs":
                num = self.num_input_steps
            elif every["what"] == "batches":
                num = self.num_batch_steps
            elif every["what"] == "epochs":
                num = self.epoch
            else:
                continue

            if num - every["last_num"] >= every["num"]:
                every["callable"]()
                every["last_num"] = num
        self._train_auxiliary_time += time.time() - start_time_aux

    @classmethod
    def from_dict(cls, data: dict):
        kwargs = cls.get_kwargs_from_dict(data)
        return cls(**kwargs)

    @classmethod
    def get_kwargs_from_dict(cls, data: dict) -> dict:
        from .experiment import get_trainer_kwargs_from_dict
        return get_trainer_kwargs_from_dict(data)
