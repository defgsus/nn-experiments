import math
from collections import OrderedDict
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.models.cnn import Conv2dBlock
from src.util.image import set_image_channels, image_resize_crop
from src.util import to_torch_device


class Encoder2d(nn.Module):
    """
    Abstract base class for image encoders
    """

    _registered_encoder_classes: Dict[str, Type["Encoder2d"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Encoder2d._registered_encoder_classes[cls.__name__] = cls

    def __init__(
            self,
            shape: Tuple[int, int, int],
            code_size: int,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.code_size = int(code_size)

    # ---- override these ----

    @property
    def device(self):
        """Returns the `torch.device` of the whole encoder"""
        raise NotImplementedError

    @classmethod
    def from_data(cls, data: dict):
        """
        Create new instance from a state_dict

        Use extra_data to store class instantiation variables
        """
        raise NotImplementedError

    # ---- public API ----

    def encode_image(
            self,
            image: torch.Tensor,
            requires_grad: bool = False,
            normalize: bool = False,
    ) -> torch.Tensor:
        """
        Convenience wrapper around forward().

        Automatically adjusts image size and channels if needed.

        :param image: Tensor of dim NxCxHxW
        :param requires_grad: bool, if False (default), the encoding is wrapped in `torch.no_grad()`.
        :param normalize: bool, if True, embedding is devided by it's `torch.norm`
        :return: Tensor of NxF
        """
        def _encode():
            nonlocal image
            if image.shape[-3:] != self.shape:
                image = image_resize_crop(image, self.shape[-2:])
                image = set_image_channels(image, self.shape[-3])

            embedding = self(image)
            if normalize:
                embedding_norm = embedding.norm(dim=1, keepdim=True)
                mask = (embedding_norm > 0)
                embedding[mask] = embedding[mask] / embedding_norm[mask]
            return embedding

        if requires_grad:
            return _encode()
        else:
            with torch.no_grad():
                return _encode()

    @classmethod
    def from_torch(cls, f, device: Union[None, str, torch.device] = "cpu"):
        """
        Instantiate a model from a dict or file

        :param f: dict (from `Model.state_dict`) or filename of a saved state_dict in torch format
        :param device: device to put the encoder to
        :return: new instance
        """
        device = to_torch_device(device)

        if isinstance(f, (dict, OrderedDict)):
            data = f
        else:
            data = torch.load(f, map_location=device)

        if "state_dict" in data:
            data = data["state_dict"]

        return cls.from_data(data).to(device)

    # --- internal API ---

    def get_extra_state(self):
        """
        Override this and ADD your own.
        """
        return {
            "class": self.__class__.__name__,
            "shape": self.shape,
            "code_size": self.code_size,
        }

    def set_extra_state(self, state):
        pass
