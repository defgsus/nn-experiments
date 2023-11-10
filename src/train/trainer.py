import json
import os
import re
import math
import random
import itertools
import argparse
import shutil
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
from src.util import to_torch_device
from src.util.image import signed_to_image, get_images_from_iterable
from src.models.util import get_loss_callable


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
            training_noise: float = 0.,
            loss_function: Union[str, Callable, torch.nn.Module] = "l1",
            gradient_clipping: Optional[float] = None,
            num_train_loss_steps: int = 1000,
            reset: bool = False,
            device: Union[None, str, torch.DeviceObjType] = None,
            hparams: Optional[dict] = None,
            weight_image_kwargs: Optional[dict] = None,
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
        self.hparams = hparams
        self.weight_image_kwargs = weight_image_kwargs
        self.num_inputs_between_validations = num_inputs_between_validations
        self.num_epochs_between_validations = num_epochs_between_validations
        self.num_inputs_between_checkpoints = num_inputs_between_checkpoints
        self.epoch = 0
        self.num_batch_steps = 0
        self.num_input_steps = 0
        self.tensorboard_path = Path("./runs/") / self.experiment_name
        self.checkpoint_path = Path("./checkpoints/") / self.experiment_name
        self.device = to_torch_device(device)
        self._loss_history = []
        self._loss_steps = 0

        self.model = self.model.to(self.device)

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
        """
        input, target_features = input_batch
        output_features = self.model(input)

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
            **(extra or {}),
        }, indent=2))

    def num_trainable_parameters(self) -> int:
        return sum(
            sum(math.prod(p.shape) for p in g["params"])
            for g in self.optimizers[0].param_groups
        )

    def log_scalar(self, tag: str, value):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.num_input_steps)

    def log_image(self, tag: str, image: torch.Tensor):
        try:
            self.writer.add_image(tag=tag, img_tensor=image, global_step=self.num_input_steps)
        except TypeError as e:
            warnings.warn(f"logging image `{tag}` failed, {type(e).__name__}: {e}")

    def log_embedding(self, tag: str, embedding: torch.Tensor):
        self.writer.add_embedding(tag=tag, mat=embedding, global_step=self.num_input_steps)

    def train(self):
        print(f"---- training '{self.experiment_name}' on {self.device} ----")
        print(f"trainable params: {self.num_trainable_parameters():,}")

        last_validation_step = None
        last_validation_epoch = None
        if self.num_inputs_between_validations is not None:
            last_validation_step = -self.num_inputs_between_validations
        if self.num_epochs_between_validations is not None:
            last_validation_epoch = -self.num_epochs_between_validations

        last_checkpoint_step = None
        if self.num_inputs_between_checkpoints is not None:
            last_checkpoint_step = 0

        self.model.train(True)

        self.running = True
        while self.running:

            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                break

            total = None
            try:
                total = len(self.data_loader.dataset)
            except:
                pass

            with tqdm(
                    total=total,
                    desc=f"epoch #{self.epoch}",
            ) as progress:
                for batch_idx, input_batch in enumerate(self.iter_training_batches()):
                    if not isinstance(input_batch, (tuple, list)):
                        input_batch = (input_batch, )

                    input_batch = [
                        i.to(self.device) if isinstance(i, torch.Tensor) else i
                        for i in input_batch
                    ]

                    if self.training_noise > 0.:
                        with torch.no_grad():
                            input_batch[0] = input_batch[0] + self.training_noise * torch.randn_like(input_batch[0])

                    if self.epoch == 0 and batch_idx == 0:
                        print(" BATCH", ", ".join(str(b.shape) if hasattr(b, "shape") else "?" for b in input_batch))

                    loss_result = self._train_step(tuple(input_batch))
                    if not isinstance(loss_result, dict):
                        loss_result = {"loss": loss_result}

                    progress.update(input_batch[0].shape[0])

                    self.model.zero_grad()
                    loss_result["loss"].backward()

                    if self.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(),
                            self.gradient_clipping,
                            #error_if_nonfinite=True,
                        )

                    for opt in self.optimizers:
                        opt.step()

                    for i, sched in enumerate(self.schedulers):
                        sched.step()

                        lr = sched.get_last_lr()
                        if isinstance(lr, (list, tuple)):
                            lr = lr[0]
                        self._loss_history.append({f"learnrate_{i+1}_{type(sched.optimizer).__name__}": lr})

                    self.num_batch_steps += 1
                    self.num_input_steps += input_batch[0].shape[0]

                    self._loss_history.append({
                        key: float(value)
                        for key, value in loss_result.items()
                    })
                    self._loss_steps += input_batch[0].shape[0]
                    if self._loss_steps >= self.num_train_loss_steps:
                        losses = {}
                        for entry in self._loss_history:
                            for key, value in entry.items():
                                if key not in losses:
                                    losses[key] = []
                                losses[key].append(value)
                        for key, values in losses.items():
                            self.log_scalar(f"train_{key}", sum(values) / len(values))
                        self._loss_history.clear()
                        self._loss_steps = 0

                    if self.num_inputs_between_validations is not None:
                        if self.num_input_steps - last_validation_step >= self.num_inputs_between_validations:
                            last_validation_step = self.num_input_steps
                            last_validation_epoch = self.epoch
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

            # self.save_checkpoint(f"epoch-{epoch:03d}")
            self.save_checkpoint()

            if self.num_epochs_between_validations is not None:
                if self.epoch - last_validation_epoch >= self.num_epochs_between_validations:
                    last_validation_step = self.num_input_steps
                    last_validation_epoch = self.epoch
                    self.run_validation()

            if self.running:
                self.epoch += 1

        self.run_validation()

        self.writer.flush()

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

    def run_validation(self):
        if self.validation_loader is None:
            return

        print(f"validation @ {self.num_input_steps:,}")
        with torch.no_grad():
            self.model.for_validation = True
            self.model.train(False)
            try:
                losses = []
                for validation_batch in self.iter_validation_batches():
                    if not isinstance(validation_batch, (tuple, list)):
                        validation_batch = validation_batch,

                    validation_batch = tuple(
                        i.to(self.device) if isinstance(i, torch.Tensor) else i
                        for i in validation_batch
                    )

                    loss = self.validation_step(validation_batch)
                    if isinstance(loss, dict):
                        loss = loss["loss"]
                    losses.append(loss)
                loss = torch.Tensor(losses).mean()

            finally:
                self.model.train(True)
                self.model.for_validation = False

            # print(f"\nVALIDATION loss {float(loss)}")
            self.log_scalar("validation_loss", loss)

            loss = float(loss)

            if self._best_validation_loss is None or loss < self._best_validation_loss:
                self._best_validation_loss = loss
                self.save_checkpoint("best")
                self.save_description("best", extra={"validation_loss": loss})

            self.save_weight_image()
            self._write_step()

            if self.min_loss is not None and loss <= self.min_loss:
                print(f"min_loss ({self.min_loss}) reached with validation loss {loss}")
                self.running = False

    def save_weight_image(
            self,
            max_single_shape: Iterable[int] = (512, 512),
            nrow: int = 16,
    ):
        if not hasattr(self.model, "weight_images"):
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
        self.model.train(True)
        if hasattr(self.model, "train_step"):
            return self.model.train_step(input_batch)
        else:
            return self.train_step(input_batch)

    def _write_step(self):
        if hasattr(self.model, "write_step"):
            self.model.write_step(self)
        else:
            self.write_step()

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

