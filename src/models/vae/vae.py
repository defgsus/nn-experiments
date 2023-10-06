import warnings
from typing import List, Iterable, Tuple, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.models.util import get_loss_callable


# based on https://avandekleut.github.io/vae/
class VariationalEncoder(nn.Module):
    
    def __init__(
            self, 
            encoder: nn.Module,
            encoder_dims: int, 
            latent_dims: int,
    ):
        """
        Wraps any encoder model for use with the `VariationalAutoencoder`
        
        :param encoder: torch.Module 
        :param encoder_dims: int, the output dimension (of a single sample) of the encoder model 
        :param latent_dims: int, the desired number of latent variables (per sample)
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_dims = encoder_dims
        self.linear_mu = nn.Linear(encoder_dims, latent_dims)
        self.linear_sigma = nn.Linear(encoder_dims, latent_dims)

        self.distribution = torch.distributions.Normal(0, 1)
        self.last_mu: Optional[torch.Tensor] = None
        self.last_sigma: Optional[torch.Tensor] = None

    def forward(self, x, random: bool = True):
        assert isinstance(x, torch.Tensor), f"Expected Tensor, got '{type(x).__name__}'"
        
        # move sampler to GPU
        device = self.linear_mu.weight.device
        if self.distribution.loc.device != device:
            self.distribution.loc = self.distribution.loc.to(device)
            self.distribution.scale = self.distribution.scale.to(device)

        x = self.encoder(x)
        if x.ndim < 2:
            raise RuntimeError(
                f"Expected encoder output shape (N, {self.encoder_dims}), got {x.shape}"
            )
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[-1] != self.encoder_dims:
            raise RuntimeError(
                f"Expected encoder output shape (N, {self.encoder_dims}), got {x.shape}"
            )

        self.last_mu = mu = self.linear_mu(x)
        self.last_sigma = sigma = torch.exp(self.linear_sigma(x))

        if random:
            z = mu + sigma * self.distribution.sample(mu.shape)
        else:
            z = mu

        return z

    def weight_images(self, **kwargs):
        images = []

        if isinstance(self.encoder, nn.Sequential):
            for layer in self.encoder:
                if hasattr(layer, "weight_images"):
                    images += layer.weight_images(**kwargs)
        else:
            if hasattr(self.encoder, "weight_images"):
                images += self.encoder.weight_images(**kwargs)

        return images


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE)

    Wraps any encoder and decoder models.
    The encoder itself has to be wrapped in `VariationalEncoder`.
    """
    def __init__(
            self,
            encoder: VariationalEncoder,
            decoder: nn.Module,
            reconstruction_loss: Union[str, Callable, nn.Module] = "l2",
            reconstruction_loss_weight: float = 1.,
            kl_loss_weight: float = 1.,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._reconstruction_loss = get_loss_callable(reconstruction_loss)
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kl_loss_weight = kl_loss_weight

    def forward(self, x):
        """
        Reconstruct/reproduce `x`

        :param x: Tensor of shape (N, ...) that is accepted by wrapped encoder model
        :return: Tensor, reconstruction of input
        """
        z = self.encoder(x)
        return self.decoder(z)

    def train_step(self, batch):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        recon = self.forward(x)

        if x.shape != recon.shape:
            warnings.warn(f"XXX {x.shape} {recon.shape} {getattr(self, 'for_validation', 'X')}")
        mu, sigma = self.encoder.last_mu, self.encoder.last_sigma

        loss_recon = self._reconstruction_loss(x, recon)

        #loss_kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - .5).mean()
        loss_kl = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim=1), dim=0)

        return {
            "loss": (
                    self.reconstruction_loss_weight * loss_recon
                    + self.kl_loss_weight * loss_kl
            ),
            "loss_reconstruction": loss_recon,
            "loss_kl": loss_kl,
        }

    def weight_images(self, **kwargs):
        images = []
        if hasattr(self.encoder, "weight_images"):
            images += self.encoder.weight_images(**kwargs)
        if hasattr(self.decoder, "weight_images"):
            images += self.decoder.weight_images(**kwargs)

        return images
