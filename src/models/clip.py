import dataclasses
from typing import Optional, Union, Iterable, Callable, Tuple

import torch
import torchvision.transforms as VT

from src.util.image import set_image_channels, image_resize_crop
from src.util.embedding import normalize_embedding


class ClipSingleton:

    _models = dict()

    MODEL_NAMES = [
        "ViT-B/32",
        "open_clip:hf-hub:chs20/fare4-clip",
        "open_clip:hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K",
    ]

    @dataclasses.dataclass
    class ClipModel:
        model: torch.nn.Module
        preproc: Callable
        shape: Tuple[int, int, int]
        dtype: torch.dtype
        tokenize: Callable

    @classmethod
    def get(cls, model: Optional[str] = None, device: str = "auto") -> ClipModel:
        """
        Return CLIP model and preprocessor
        :param model: str, name of clip model, defaults to "ViT-B/32"
        :param device: str, a torch device or 'auto'
        :return: ClipModel instance
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Cuda device requested but not available")

        if model is None:
            model = cls.MODEL_NAMES[0]

        key = f"{model}/{device}"

        if key not in cls._models:
            cls._models[key] = cls._create_model(model, device)

        return cls._models[key]

    @classmethod
    def encode_image(
            cls,
            image_batch: Union[torch.Tensor, Iterable[torch.Tensor]],
            model: Optional[str] = None,
            device: str = "auto",
            interpolation: VT.InterpolationMode = VT.InterpolationMode.NEAREST,
            requires_grad: bool = False,
            normalize: bool = False,
    ):
        model = cls.get(model, device)

        if not isinstance(image_batch, torch.Tensor):
            image_batch = torch.concat([
                i.unsqueeze(0)
                for i in image_batch
            ])

        original_dtype = image_batch.dtype
        if not image_batch.dtype == model.dtype:
            image_batch = image_batch.to(model.dtype)

        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)

        if image_batch.ndim != 4:
            raise ValueError(f"Expecting image_batch.ndim == 3 or 4, got '{image_batch.ndim}'")

        if image_batch.shape[-3] != model.shape[0]:
            image_batch = set_image_channels(image_batch, model.shape[0])

        if image_batch.shape[-2:] != model.shape[-2:]:
            image_batch = image_resize_crop(image_batch, model.shape[-2:], interpolation=interpolation)

        image_batch = model.preproc(image_batch)

        device = image_batch.device
        model_device = cls._get_model_device(model.model)

        if requires_grad:
            feature_batch = model.model.encode_image(image_batch.to(model_device))
            if normalize:
                feature_batch = normalize_embedding(feature_batch)

        else:
            with torch.no_grad():
                feature_batch = model.model.encode_image(image_batch.to(model_device))
                if normalize:
                    feature_batch = normalize_embedding(feature_batch)

        # if not feature_batch.dtype == original_dtype:
        #     feature_batch = feature_batch.to(original_dtype)

        return feature_batch

    @classmethod
    def encode_text(
            cls,
            text: Union[str, Iterable[str]],
            truncate: bool = False,
            model: Optional[str] = None,
            device: str = "auto",
            requires_grad: bool = False,
            normalize: bool = False,
    ):
        model = cls.get(model, device)

        tokens = model.tokenize(text).to(cls._get_model_device(model.model))

        if requires_grad:
            embedding = model.model.encode_text(tokens)
            if normalize:
                embedding = normalize_embedding(embedding)
            return embedding
        else:
            with torch.no_grad():
                embedding = model.model.encode_text(tokens)
                if normalize:
                    embedding = normalize_embedding(embedding)
                return embedding

    @classmethod
    def _get_model_device(cls, model: torch.nn.Module) -> torch.device:
        for p in model.parameters():
            return p.device

    @classmethod
    def _create_model(cls, model: str, device: str):
        print(f"ClipSingleton: loading model '{model}' for device '{device}'")

        if model.startswith("open_clip:"):
            import open_clip
            model = model[10:]
            model, _, preproc = open_clip.create_model_and_transforms(model, device=device)
            return cls.ClipModel(
                model=model.to(torch.half),
                preproc=preproc.transforms[-1],
                shape=(3, 224, 224),
                dtype=torch.half,
                tokenize=open_clip.tokenize,
            )


        # orginal CLIP
        import clip
        model, preproc = clip.load(name=model, device=device)
        return cls.ClipModel(
            model=model,
            preproc=preproc.transforms[-1],
            shape=(3, 224, 224),
            dtype=torch.half,
            tokenize=clip.tokenize,
        )


if __name__ == "__main__":
    text_emb = ClipSingleton.encode_text(
        "White noise", model=ClipSingleton.MODEL_NAMES[-2]
    )
    print("text_emb ", text_emb.shape, text_emb.dtype)

    image_emb = ClipSingleton.encode_image(
        torch.rand(1, 3, 100, 100), model=ClipSingleton.MODEL_NAMES[-1]
    )
    print("image_emb", image_emb.shape, image_emb.dtype)

