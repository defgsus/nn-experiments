import sys

import torch
import torch.nn as nn


class DebugLayer(nn.Module):

    def __init__(self, name: str = "debug"):
        super().__init__()
        self.name = name

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]

        reps = []
        for i in x:
            rep = type(i).__name__
            if isinstance(i, torch.Tensor):
                rep = f"Tensor(shape={tuple(i.shape)}, dtype={i.dtype})"
            reps.append(rep)

        rep = type(x).__name__
        if len(reps) == 1:
            rep = reps[0]
        elif len(reps) > 1:
            rep = ", ".join(reps)

        print(f"DebugLayer(name={self.name}): input: {rep}", file=sys.stderr)
        return x
