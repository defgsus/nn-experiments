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


class Encoder1d(nn.Module):
    """
    Abstract base class for audio encoders
    """

    _registered_encoder_classes: Dict[str, Type["Encoder1d"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Encoder1d._registered_encoder_classes[cls.__name__] = cls

    def __init__(
            self,
            shape: Tuple[int, int],
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

    def encode_audio(
            self,
            wave: torch.Tensor,
            requires_grad: bool = False,
            normalize: bool = False,
    ) -> torch.Tensor:
        """
        Convenience wrapper around forward().

        Automatically adjusts channels if needed.

        :param wave: Tensor of dim NxCxS
        :param requires_grad: bool, if False (default), the encoding is wrapped in `torch.no_grad()`.
        :param normalize: bool, if True, embedding is divided by its `torch.norm`
        :return: Tensor of NxF
        """
        def _encode():
            nonlocal wave
            if wave.shape[-2:] != self.shape:
                wave = set_image_channels(wave, self.shape[-3])

            embedding = self(wave)
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
