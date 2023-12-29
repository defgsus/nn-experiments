import torch
import torch.nn.functional as F


class Ca1AdditiveRules:
    def __init__(
            self,
            num_states: int = 2,
            num_neighbours: int = 1,
    ):
        self.num_states = num_states
        self.num_neighbours = num_neighbours
        self.kernel_size = 1 + 2 * num_neighbours
        self.num_rules = num_states ** self.kernel_size

    def __len__(self):
        return self.num_rules

    def kernel(self, rule: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        kernel = [
            (rule // (self.num_states ** k)) % self.num_states
            for k in range(self.kernel_size)
        ]
        return torch.Tensor(kernel).to(dtype)


def ca1_additive_step(
        input: torch.Tensor,
        kernel: torch.Tensor,
        num_states: int,
        iterations: int,
        wrap: bool = False,
) -> torch.Tensor:
    if input.ndim != 1:
        raise ValueError(f"Expected input.ndim=1, got {input.shape}")
    if kernel.ndim != 1:
        raise ValueError(f"Expected kernel.ndim=1, got {kernel.shape}")

    num_n = (kernel.shape[-1] - 1) / 2
    if num_n != int(num_n):
        raise ValueError(f"kernel size must be `num-neighbours * 2 + 1`, got {kernel.shape[-1]}")
    num_n = int(num_n)

    kernel = kernel.view(1, 1, kernel.shape[-1])
    state = input.view(1, input.shape[-1])

    history = [state]
    for it in range(iterations):

        if wrap:
            state = torch.concat([state[:, -num_n:], state, state[:, :num_n]], dim=-1)

        state = F.conv1d(state, kernel, padding=num_n if not wrap else 0) % num_states
        history.append(state)

    return torch.concat(history)
