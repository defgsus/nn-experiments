import torch


def normalize_embedding(embedding: torch.Tensor, save: bool = True):
    if embedding.ndim == 1:
        return normalize_embedding(embedding.unsqueeze(0), save=save).squeeze(0)

    assert embedding.ndim == 2, f"Got {embedding.ndim}"

    embedding_norm = embedding.norm(dim=1, keepdim=True)
    embedding = embedding / embedding_norm
    if save:
        with torch.no_grad():
            embedding[embedding.isnan()] = 0.

    return embedding
