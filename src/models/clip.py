from typing import Optional, Union, Iterable

import torch
import torchvision.transforms as VT
import clip

from src.util.image import set_image_channels, image_resize_crop


class ClipSingleton:

    _models = dict()

    @classmethod
    def get(cls, model: Optional[str] = None, device: str = "auto") -> tuple:
        """
        Return CLIP model and preprocessor
        :param model: str, name of clip model, defaults to "ViT-B/32"
        :param device: str, a torch device or 'auto'
        :return: tuple of (Module, Module)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Cuda device requested but not available")

        if model is None:
            model = "ViT-B/32"

        key = f"{model}/{device}"

        if key not in cls._models:
            cls._models[key] = clip.load(name=model, device=device)

        return cls._models[key]

    @classmethod
    def encode_image(
            cls,
            image_batch: torch.Tensor,
            model: Optional[str] = None,
            device: str = "auto",
            interpolation: VT.InterpolationMode = VT.InterpolationMode.NEAREST,
            requires_grad: bool = False,
    ):
        model, preproc = cls.get(model, device)

        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)

        if image_batch.ndim != 4:
            raise ValueError(f"Expecting image_batch.ndim == 3 or 4, got '{image_batch.ndim}'")

        clip_shape = (3, 244, 244)

        if image_batch.shape[-3] != clip_shape[0]:
            image_batch = set_image_channels(image_batch, 3)

        if image_batch.shape[-2:] != clip_shape[-2:]:
            image_batch = image_resize_crop(image_batch, clip_shape[-2:], interpolation=interpolation)

        image_batch = preproc.transforms[-1](image_batch)

        device = image_batch.device
        model_device = cls._get_model_device(model)

        if requires_grad:
            feature_batch = model.encode_image(image_batch.to(model_device))
        else:
            with torch.no_grad():
                feature_batch = model.encode_image(image_batch.to(model_device))

        return feature_batch.to(device).to(image_batch.dtype)

    @classmethod
    def encode_text(
            cls,
            text: Union[str, Iterable[str]],
            model: Optional[str] = None,
            device: str = "auto",
            requires_grad: bool = False,
    ):
        model, preproc = cls.get(model, device)

        tokens = clip.tokenize(text).to(cls._get_model_device(model))

        if requires_grad:
            return model.encode_text(tokens)
        else:
            with torch.no_grad():
                return model.encode_text(tokens)

    @classmethod
    def _get_model_device(cls, model: torch.nn.Module) -> torch.device:
        for p in model.parameters():
            return p.device