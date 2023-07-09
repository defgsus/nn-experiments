import math
from typing import Optional, Callable, List, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


class RBM(nn.Module):
    def __init__(
            self,
            num_in: int,
            num_out: int,
            act_fn: Optional[Callable] = torch.sigmoid,
    ):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.act_fn = act_fn

        self.bias_visible = nn.Parameter(torch.randn(1, self.num_in))
        self.bias_hidden = nn.Parameter(torch.randn(1, self.num_out))
        self.weight = nn.Parameter(torch.randn(self.num_out, self.num_in) / math.sqrt(self.num_in + self.num_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_in)
        y = F.linear(x, self.weight, self.bias_hidden)
        if self.act_fn is not None:
            y = self.act_fn(y)
        return y

    def visible_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_in)
        y = F.linear(x, self.weight, self.bias_hidden)
        if self.act_fn is not None:
            y = self.act_fn(y)
        return y.bernoulli()

    def hidden_to_visible(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_out)
        y = F.linear(x, self.weight.t(), self.bias_visible)
        if self.act_fn is not None:
            y = self.act_fn(y)
        return y.bernoulli()

    def contrastive_divergence(
            self,
            x: torch.Tensor,
            num_steps: int = 2,
            noise_level: float = 0.,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_first = x_last = x.view(-1, self.num_in)

        if noise_level:
            state = self.visible_to_hidden(x_first + torch.randn_like(x_first) * noise_level)
        else:
            state = self.visible_to_hidden(x_first)

        for step in range(num_steps):
            x_last = self.hidden_to_visible(state)
            if step < num_steps - 1:
                state = self.visible_to_hidden(x_last)
        return x_first, x_last

    def gibbs_sample(
            self,
            x: torch.Tensor,
            num_steps: int = 2,
    ) -> torch.Tensor:
        samples = []
        state = self.visible_to_hidden(x)
        for step in range(num_steps):
            state = self.hidden_to_visible(state)
            samples.append(state.unsqueeze(0))
            if step < num_steps - 1:
                state = self.visible_to_hidden(state)
        return torch.cat(samples).mean(0)

    def weight_images(self, shape: Optional[Tuple[int, ...]] = None, max_nodes: int = 128) -> List[torch.Tensor]:
        return [
            (self.weight[i].view(*shape) if shape is not None else self.weight[i])
            for i in range(min(max_nodes, self.num_out))
        ]

    def free_energy(self, x):
        v_term = torch.matmul(x, self.bias_visible.t())
        w_x_h = F.linear(x, self.weight, self.bias_hidden)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def train_step(self, input_batch, noise_level: float = 0.3) -> torch.Tensor:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        first_state, last_state = self.contrastive_divergence(input_batch, noise_level=noise_level)

        loss = (self.free_energy(first_state) - self.free_energy(last_state)) / self.num_out
        #loss = F.l1_loss(first_state, last_state)

        #if getattr(self, "for_validation", None):
        #    print(first_state, last_state)
        #    print(first_state.min(), first_state.max(), last_state.min(), last_state.max())
        #    print(loss, first_state.shape)

        return loss

    def duplicate_output_cells(self, noise_level: float = 0.02):
        self.weight = nn.Parameter(
            .5 * torch.concat([self.weight, self.weight + torch.randn_like(self.weight) * noise_level])
        )

        self.bias_hidden = nn.Parameter(
            torch.concat([self.bias_hidden, self.bias_hidden + torch.randn_like(self.bias_hidden) * noise_level])
            .reshape(1, -1)
        )

        self.num_out *= 2
