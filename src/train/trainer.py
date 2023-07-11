import json
import os
import re
import math
import random
import itertools
import argparse
import shutil
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util.image import signed_to_image, get_images_from_iterable


class Trainer:

    def __init__(
            self,
            experiment_name: str,
            model: torch.nn.Module,
            data_loader: DataLoader,
            validation_loader: Optional[DataLoader] = None,
            min_loss: Optional[float] = None,
            max_epoch: Optional[int] = None,
            max_inputs: Optional[int] = 10_000_000,
            optimizers: Iterable[torch.optim.Optimizer] = tuple(),
            num_inputs_between_validations: int = 3_000,
            reset: bool = False,
            device: Optional[Union[str, torch.DeviceObjType]] = None,
            hparams: Optional[dict] = None,
            weight_image_kwargs: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.model = model
        self.data_loader = data_loader
        self.validation_loader = validation_loader
        self._validation_batch: Optional[torch.Tensor] = None
        self.min_loss = min_loss
        self.max_epoch = max_epoch
        self.max_inputs = max_inputs
        self.optimizers = list(optimizers)
        self.hparams = hparams
        self.weight_image_kwargs = weight_image_kwargs
        self.num_inputs_between_validations = num_inputs_between_validations
        self.epoch = 0
        self.num_batch_steps = 0
        self.num_input_steps = 0
        self.tensorboard_path = Path("./runs/") / self.experiment_name
        self.checkpoint_path = Path("./checkpoints/") / self.experiment_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        if reset:
            if self.tensorboard_path.exists():
                shutil.rmtree(self.tensorboard_path)
            if self.checkpoint_path.exists():
                shutil.rmtree(self.checkpoint_path)
        self.writer = SummaryWriter(str(self.tensorboard_path))

        self._every_callbacks = []
        self._setup_every_callbacks()

    def train_step(self, input_batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, input_batch) -> torch.Tensor:
        return self._train_step(input_batch)

    def write_step(self):
        pass

    def load_checkpoint(self, name: str = "snapshot"):
        checkpoint_filename = self.checkpoint_path / f"{name}.pt"

        if checkpoint_filename.exists():
            print(f"loading {checkpoint_filename}")
            checkpoint_data = torch.load(checkpoint_filename)
            self.model.load_state_dict(checkpoint_data["state_dict"])
            self.epoch = checkpoint_data.get("epoch") or 0
            self.num_batch_steps = checkpoint_data.get("num_batch_steps") or 0
            self.num_input_steps = checkpoint_data.get("num_input_steps") or 0

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
            },
            checkpoint_filename,
        )

    def save_description(self, name: str = "description"):
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
        }, indent=2))

    def num_trainable_parameters(self) -> int:
        return sum(
            sum(math.prod(p.shape) for p in g["params"])
            for g in self.optimizers[0].param_groups
        )

    def log_scalar(self, tag: str, value):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.num_input_steps)

    def log_image(self, tag: str, image: torch.Tensor):
        self.writer.add_image(tag=tag, img_tensor=image, global_step=self.num_input_steps)

    def train(self):
        print(f"---- training on {self.device}")
        print(f"trainable params: {self.num_trainable_parameters():,}")

        last_validation_step = -self.num_inputs_between_validations
        self.running = True
        while self.running:
            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                break

            for batch_idx, input_batch in enumerate(tqdm(self.data_loader, desc=f"epoch #{self.epoch}")):
                if isinstance(input_batch, (tuple, list)):
                    input_batch = input_batch[0]
                input_batch = input_batch.to(self.device)

                if self.epoch == 0 and batch_idx == 0:
                    print("BATCH", input_batch.shape)

                loss = self._train_step(input_batch)

                self.model.zero_grad()
                loss.backward()
                for opt in self.optimizers:
                    opt.step()

                self.num_batch_steps += 1
                self.num_input_steps += input_batch.shape[0]

                self.log_scalar("train_loss", loss)

                if self.num_input_steps - last_validation_step > self.num_inputs_between_validations:
                    last_validation_step = self.num_input_steps
                    self.run_validation()

                # print(f"\nepoch {self.epoch}, batch {self.num_batch_steps}, image {self.num_input_steps}, loss {float(loss)}")

                self._run_every_callbacks()

                if self.max_inputs and self.num_input_steps >= self.max_inputs:
                    print(f"max_inputs ({self.max_inputs}) reached")
                    self.running = False
                    break

            # self.save_checkpoint(f"epoch-{epoch:03d}")
            self.save_checkpoint("snapshot")

            if self.running:
                self.epoch += 1

        self.run_validation()

        self.writer.flush()

    @property
    def validation_batch(self) -> Optional[torch.Tensor]:
        if self.validation_loader is None:
            return
        if self._validation_batch is None:
            self._validation_batch = torch.cat([
                i[0] if isinstance(i, (tuple, list)) else i
                for i in self.validation_loader
            ]).to(self.device)
        return self._validation_batch

    def run_validation(self):
        if self.validation_loader is None:
            return

        print(f"validation @ {self.num_input_steps}")
        with torch.no_grad():
            self.model.for_validation = True
            loss = self.validation_step(self.validation_batch)
            self.model.for_validation = False

            # print(f"\nVALIDATION loss {float(loss)}")
            self.log_scalar("validation_loss", loss)

            self.save_weight_image()
            self.write_step()

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

        images = self.model.weight_images(**(self.weight_image_kwargs or {}))
        max_shape = None
        max_single_shape = list(max_single_shape)
        for image_idx, image in enumerate(images):

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

            images[image_idx] = signed_to_image(images[image_idx])

        self.log_image("weights", make_grid(images, nrow=nrow))

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

    def _train_step(self, input_batch: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "train_step"):
            return self.model.train_step(input_batch)
        else:
            return self.train_step(input_batch)

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

