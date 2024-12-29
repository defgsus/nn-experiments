from typing import Union, Tuple

import numpy as np
import torch


def normalize_embedding(embedding: Union[torch.Tensor, np.ndarray], save: bool = True):
    if embedding.ndim == 1:
        return normalize_embedding(embedding[None, ...], save=save)[0]

    assert embedding.ndim == 2, f"Got {embedding.ndim}"

    if isinstance(embedding, np.ndarray):
        embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / embedding_norm
        if save:
            embedding[np.isnan(embedding)] = 0.

    else:

        embedding_norm = embedding.norm(dim=1, keepdim=True)
        embedding = embedding / embedding_norm
        if save:
            with torch.no_grad():
                embedding[embedding.isnan()] = 0.

    return embedding


def create_diagonal_matrix(shape: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape, shape)
    else:
        assert len(shape) == 2, f"Got {shape}"

    if shape[-2] < shape[-1]:
        return create_diagonal_matrix((shape[-1], shape[-2])).T

    x_range = torch.arange(0, shape[-1]).float()
    y_range = torch.linspace(0, shape[-1] - 1, shape[-2])
    m_x = x_range.unsqueeze(0).repeat(shape[-2], 1)
    m_y = y_range.unsqueeze(0).repeat(shape[-1], 1)
    m = 1 - (m_x - m_y.T).abs().clamp(0, 1)
    return m #/ torch.norm(m, dim=-1, keepdim=True)
