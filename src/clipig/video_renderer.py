import io
import os
import yaml
import shutil
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
from tqdm import tqdm

from ..util import to_torch_device
from ..util.image import image_minimum_size
from .parameters import get_complete_clipig_task_config
from .clipig_task import ClipigTask


class ClipigVideoRenderer:

    def __init__(
            self,
            config: Union[dict, str, Path],
            fps: int = 30,
            video_frame_stride: int = 1,
            transformation_frame_stride: Optional[int] = None,
            store_directory: Optional[Union[str, Path]] = None,
            display_jupyter: bool = False,
    ):
        if not isinstance(config, dict):
            fp = io.StringIO(config)
            config = yaml.safe_load(fp)

        self.config = get_complete_clipig_task_config(config)
        self.store_directory = Path(store_directory) if store_directory is not None else None
        self.fps = fps
        self.video_frame_stride = video_frame_stride
        self.transformation_frame_stride = video_frame_stride if transformation_frame_stride is None else transformation_frame_stride
        self.display_jupyter = display_jupyter

        self.clipig_frame = 0
        self.video_frame = 0
        self.task: Optional[ClipigTask] = None

        if self.display_jupyter:
            from IPython.display import display
            import ipywidgets
            from src.util.widgets import ImageWidget
            self._image_widget = ImageWidget()
            self._status_widget = ipywidgets.Text()
            display(self._image_widget)
            display(self._status_widget)

    @property
    def second(self) -> float:
        return self.clipig_frame / self.video_frame_stride / self.fps

    def transform(self, pixels: torch.Tensor, delta: float):
        return pixels

    def post_process(self, pixels: torch.Tensor, delta: float):
        return pixels

    def run(
            self,
            seconds: float,
            reset: bool = False,
    ):
        num_iterations = seconds * self.video_frame_stride * self.fps

        config = deepcopy(self.config)
        config["num_iterations"] = num_iterations
        config["pixel_yield_delay_sec"] = 0.

        image_idx = 0
        frame_idx = 0

        if self.store_directory is not None:
            if self.store_directory.exists():
                if reset:
                    shutil.rmtree(self.store_directory)
                else:
                    filenames = sorted(self.store_directory.glob("*.png"))
                    if filenames:
                        frame_idx = len(filenames)
                        image_idx = frame_idx * self.video_frame_stride
                        config["initialize"] = "input"
                        config["input_image"] = VF.to_tensor(PIL.Image.open(str(filenames[-1])))

            os.makedirs(self.store_directory, exist_ok=True)

        self.video_frame = frame_idx
        self.clipig_frame = image_idx
        self.task = ClipigTask(config)
        status = "requested"

        last_video_frame = self.clipig_frame
        last_transformation_frame = self.clipig_frame
        try:
            with tqdm(total=num_iterations) as progress:
                for event in self.task.run():
                    if "status" in event:
                        status = event["status"]

                    if "pixels" in event:
                        progress.update(1)
                        clipig_frame = self.clipig_frame + 1
                        pixels = event["pixels"].clamp(0, 1)

                        if clipig_frame - last_transformation_frame >= self.transformation_frame_stride:
                            delta = (clipig_frame - last_transformation_frame) / self.video_frame_stride / self.fps
                            last_transformation_frame = clipig_frame

                            with torch.no_grad():
                                pixels = self.transform(pixels, delta).clamp(0, 1)
                                self.task.source_model.set_image(pixels)

                        if clipig_frame - last_video_frame >= self.video_frame_stride:
                            delta = (clipig_frame - last_video_frame) / self.video_frame_stride / self.fps
                            last_video_frame = clipig_frame
                            with torch.no_grad():
                                pixels = self.post_process(pixels, delta).clamp(0, 1)
                                # self.task.source_model.set_image(pixels)

                            if self.store_directory is not None or self.display_jupyter:
                                pixels_pil = VF.to_pil_image(pixels)
                                if self.store_directory is not None:
                                    pixels_pil.save(self.store_directory / f"frame-{frame_idx:08}.png")

                                if self.display_jupyter:
                                    self._image_widget.set_pil(image_minimum_size(pixels_pil, width=500))

                            self.video_frame += 1

                        self.clipig_frame += 1

                    if self.display_jupyter:
                        self._status_widget.value = (
                            f"status: {status}"
                            f", second={self.second:.2f}"
                            f", video_frame={self.video_frame}, clipg_frame={self.clipig_frame}"

                        )

        except KeyboardInterrupt:
            print("stopped")
            pass
