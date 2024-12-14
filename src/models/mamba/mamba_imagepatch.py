from typing import Tuple

from .mamba import *


class MambaImagePatch(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            args: ModelArgs,
    ):
        super().__init__()
        self.shape = shape
        self.args = args

        self.proj_in = nn.Linear(math.prod(shape), args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.proj_out = nn.Linear(args.d_model, math.prod(shape))

    def forward(self, input: torch.Tensor):
        """
        :param input: Tensor of shape [B, L, C, H, W]
        """
        # print("IN", input.shape)
        x = self.proj_in(input.flatten(2))
        # print("X", x.shape)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        out = self.proj_out(x)
        return out.view(input.shape)

