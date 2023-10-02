import math
from typing import Optional, Callable, List, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.util.embedding import normalize_embedding

class RBM(nn.Module):
    def __init__(
            self,
            num_in: int,
            num_out: int,
            act_fn: Optional[Callable] = torch.sigmoid,
            dropout: float = 0.,
            train_max_similarity: float = 0.,
    ):
        """
        Restricted Boltzmann Machine

        :param num_in: int, number of inputs
        :param num_out: int, number of outputs
        :param act_fn: callable, activation function, default is sigmoid.
            This should typically produce values in the range [0, 1]
        :param dropout: float,
            dropout ratio used in training (typically around 0.5 if used)
        :param train_max_similarity: float,
            If defined will add the following training constraint:
            The average cosine similarity between all embeddings of a batch of inputs
            should not be greater than this value. 0.5 may be a good choice.
            This will drive dissimilarity between the resulting embeddings.
            Although, using this depends on:
                1. a large batch size
                2. a dataset that contains enough dissimilar inputs per batch
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.act_fn = act_fn
        self.train_max_similarity = train_max_similarity
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

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
        y = self.forward(x).bernoulli()
        if self.dropout is not None:
            y = self.dropout(y)
        return y

    def hidden_to_visible(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_out)
        y = F.linear(x, self.weight.t(), self.bias_visible)
        if self.act_fn is not None:
            y = self.act_fn(y)
        y = y.bernoulli()
        if self.dropout is not None:
            y = self.dropout(y)
        return y

    def contrastive_divergence(
            self,
            x: torch.Tensor,
            num_steps: int = 2,
            noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_first = x_last = x.view(-1, self.num_in)

        if noise_std:
            state = self.visible_to_hidden(x_first + torch.randn_like(x_first) * noise_std)
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

    def train_step(self, input_batch, noise_std: float = 0.0):
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        first_state, last_state = self.contrastive_divergence(input_batch, noise_std=noise_std)

        energy_loss = (self.free_energy(first_state) - self.free_energy(last_state)) / self.num_out
        energy_loss = energy_loss.clamp(-100, 100)

        if self.train_max_similarity <= 0.:
            return energy_loss

        else:
            embeddings = normalize_embedding(self.forward(input_batch))
            sim = embeddings @ embeddings.T
            sim_loss = (sim.mean() - self.train_max_similarity).clamp_min(0.)

            return {
                "loss": energy_loss + sim_loss,
                "loss_energy": energy_loss,
                "loss_similarity": sim_loss,
            }

    def duplicate_output_cells(self, noise_level: float = 0.02):
        self.weight = nn.Parameter(
            .5 * torch.concat([self.weight, self.weight + torch.randn_like(self.weight) * noise_level])
        )

        self.bias_hidden = nn.Parameter(
            torch.concat([self.bias_hidden, self.bias_hidden + torch.randn_like(self.bias_hidden) * noise_level])
            .reshape(1, -1)
        )

        self.num_out *= 2
