import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    Simple wrapper to contain encoder/decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def weight_images(self, **kwargs):
        images = []
        if hasattr(self.encoder, "weight_images"):
            images.extend(self.encoder.weight_images(**kwargs))
        if hasattr(self.decoder, "weight_images"):
            images.extend(self.decoder.weight_images(**kwargs))
        return images
