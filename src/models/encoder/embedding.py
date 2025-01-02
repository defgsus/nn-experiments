import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util.embedding import create_diagonal_matrix


class DiagonalEmbedding(nn.Module):

    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            diagonal: bool = True,
            symmetric: bool = True,
            fft: bool = False,
            fft_concat_dim: int = -1,
    ):
        """
        Wrapper around torch.nn.Embedding

        :param channels_in: int, vocabulary size
        :param channels_out: int, internal representation size
        :param diagonal: bool, if True, the embedding weights are initialized with
            a diagonal matrix, e.g. if channels_in==channels_out, the representation
            matches the input
        :param symmetric: bool, if True, the embedding weights and readout weights are shared.
            If False, the readout has its own set of weights.
        :param fft: bool, If True, the representation is the concatenation of the
            real and imaginary FFT transform
        :param fft_concat_dim: int, either -1 or -2,
            if -1, fft real and imaginary output is concatenated along the sequence dimension
            if -2, it's concatenated along the channel dimensions and `channels_out` is divided by 2!
        """
        super().__init__()
        self._diagonal = diagonal
        self._symmetric = symmetric
        self._fft = fft
        self._fft_concat_dim = fft_concat_dim

        if fft:
            assert sum(int(x) for x in list(bin(channels_out)[2:])) == 1, \
                f"`channels_in` must be a power of 2 when using `fft`"
            assert fft_concat_dim in (-1, -2), f"`fft_concat_dim` must be -1 or -2, got '{fft_concat_dim}'"
            if fft_concat_dim == -2:
                channels_out //= 2

        self.input = nn.Embedding(channels_in, channels_out)
        with torch.no_grad():
            if diagonal:
                self.input.weight[:] = create_diagonal_matrix(self.input.weight.shape)
            elif fft:
                self.input.weight[:] = F.softmax(self.input.weight, dim=-1)

        if not symmetric:
            self.output = nn.Linear(channels_out, channels_in)

    def extra_repr(self) -> str:
        return (
            f"diagonal={self._diagonal}, symmetric={self._symmetric}"
            f", fft={self._fft}, fft_concat_dim={self._fft_concat_dim}"
        )

    def forward(
            self,
            x: torch.Tensor,
            reverse: bool = False,
    ) -> torch.Tensor:
        """
        Converts token indices to representation or representation to token class logits

        :param x: torch.Tensor,
            if reverse==False, the token indices of shape [B, L] (where L is sequence length),
            if reverse==True and fft==False, the representation of shape [B, C, L]
            if reverse==True and fft==True, the representation
                of shape [B, C, L] if `fft_concat_dim==-2` or [B, C, L*2] if `fft_concat_dim==-1`
        :param reverse: bool, if True, reverses the embedding
        :return: torch.Tensor,
            if reverse==False and fft==False, the representation of shape [B, C, L]
            if reverse==False and fft==True, the representation
                of shape [B, C, L] if `fft_concat_dim==-2` or [B, C, L*2] if `fft_concat_dim==-1`
            if reverse==True, the token class logits of shape [B, L, V] (where V is vocab_size)
        """
        if not reverse:

            outp = self.input(x).permute(0, 2, 1)
            if self._fft:
                outp = torch.fft.fft(outp, dim=-2)
                outp = torch.concat([outp.real, outp.imag], dim=self._fft_concat_dim)
            return outp

        else:
            if self._fft:
                x = torch.complex(*torch.split(x, x.shape[self._fft_concat_dim] // 2, dim=self._fft_concat_dim))
                x = torch.fft.ifft(x, dim=-2).real

            if self._symmetric:
                return (self.input.weight @ x).permute(0, 2, 1).contiguous()
            else:
                return self.output(x.permute(0, 2, 1))
