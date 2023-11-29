from typing import Union

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
