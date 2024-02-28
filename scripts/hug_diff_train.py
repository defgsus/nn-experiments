import os
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Optional, Callable, Union

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import TensorDataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader

import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from src.util.module import num_module_parameters
from src.datasets import *
from experiments.datasets import PixelartDataset


@dataclass
class TrainingConfig:

    image_shape = (4, 32, 32)
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 5

    dataset = PixelartDataset(shape=image_shape)
    num_classes = len(PixelartDataset.LABELS)
    num_train_timesteps = 1000

    mixed_precision = "no"# "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-test-01/03"  #

    overwrite_output_dir = True

    seed = 0


def create_model(config: TrainingConfig):

    model = diffusers.UNet2DModel(
        sample_size=config.image_shape[-1],  # the target image resolution
        in_channels=config.image_shape[0],  # the number of input channels, 3 for RGB images
        out_channels=config.image_shape[0],  # the number of output channels
        num_class_embeds=config.num_classes,

        layers_per_block=2,  # how many ResNet layers to use per UNet block
        #block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        block_out_channels=(128, 128, 256, 512),

        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            #"DownBlock2D",
            #"DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),

        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            #"UpBlock2D",
            #"UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


class DDPMPipelineWithClasses(diffusers.DDPMPipeline):

    @dataclass
    class CallbackArg:
        pipeline: "DDPMPipelineWithClasses"
        image: torch.Tensor
        iteration: int
        timestep: int

    @torch.no_grad()
    def __call__(
            self,
            classes: Iterable[int] = (0,),
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "pt",
            return_dict: bool = True,
            callback: Optional[Callable[[CallbackArg], torch.Tensor]] = None,
    ) -> Union[diffusers.ImagePipelineOutput, Tuple]:
        from diffusers.utils.torch_utils import randn_tensor

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        class_labels = list(classes)
        while len(class_labels) < batch_size:
            class_labels.extend(class_labels)
        class_labels = torch.LongTensor(class_labels[:batch_size]).to(self.device)

        return self.run_inference_on(
            image=image / 2. + .5,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_dict=return_dict,
            generator=generator,
            callback=callback,
        )

    @torch.no_grad()
    def run_inference_on(
            self,
            image: torch.Tensor,
            class_labels: Union[int, Iterable, torch.LongTensor],
            num_inference_steps: int = 1000,
            timestep_offset: int = 0,
            timestep_count: Optional[int] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pt",
            return_dict: bool = True,
            callback: Optional[Callable[[CallbackArg], torch.Tensor]] = None,
    ):
        image = image.to(self.device)

        if image.ndim == 4:
            pass
        elif image.ndim == 3:
            image = image.unsqueeze(0)
        else:
            raise ValueError(f"`image` must have 4 (or 3) dimensions, got {image.shape}")

        image = image * 2. - 1.

        if isinstance(class_labels, int):
            class_labels = torch.LongTensor([class_labels]).to(self.device)
        elif not isinstance(class_labels, torch.Tensor):
            class_labels = torch.LongTensor(class_labels).to(self.device)

        if class_labels.shape[0] != image.shape[0]:
            raise ValueError(f"batch-size of `class_labels` must match `image`, expected {image.shape[0]}, got {class_labels.shape}")

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps = timestepsX = self.scheduler.timesteps[timestep_offset:]
        if timestep_count is not None:
            timesteps = timesteps[:timestep_count]
        # print(timestep_count, timestep_offset, timesteps, timestepsX)
        for idx, t in enumerate(self.progress_bar(timesteps)):
            # 1. predict noise model_output
            model_output = self.unet(image, t, class_labels).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            if callback is not None:
                image = callback(self.CallbackArg(
                    pipeline=self, image=image, iteration=idx, timestep=t
                ))

        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            pass
        else:
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return diffusers.ImagePipelineOutput(images=image)


def main():
    config = TrainingConfig()

    model = create_model(config)
    print(model)
    print(f"params: {num_module_parameters(model):,}")

    transforms = [Normalize([0.5], [0.5])]
    dataset = TransformDataset(config.dataset, transforms=transforms)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    def evaluate(config, epoch, pipeline: DDPMPipelineWithClasses):
        from diffusers.utils import make_image_grid

        for inference_steps in sorted({20, 100, config.num_train_timesteps}):
            images = pipeline(
                classes=range(min(config.num_classes, config.eval_batch_size)),
                batch_size=config.eval_batch_size,
                generator=torch.manual_seed(config.seed),
                num_inference_steps=inference_steps,
                output_type="pil",
            ).images

            image_grid = make_image_grid(images, rows=4, cols=4)

            test_dir = Path(config.output_dir) / "samples"
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save(f"{test_dir}/{epoch:04d}-{inference_steps}.png")

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader.dataset), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images, class_labels = batch
            #if isinstance(batch, (tuple, list)):
            #    clean_images = batch[0]

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # print("X", noisy_images.shape, timesteps.shape, class_labels.shape)
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(bs)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += bs

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipelineWithClasses(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
