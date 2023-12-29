import torch
import torch.nn.functional as F


class Ca1ReplaceRules:
    def __init__(
            self,
            num_states: int = 2,
            num_neighbours: int = 1,
    ):
        self.num_states = num_states
        self.num_neighbours = num_neighbours
        self.kernel_size = 1 + 2 * num_neighbours
        self.num_rules = num_states ** num_states ** self.kernel_size

    def __len__(self):
        return self.num_rules

    def lookup(self, rule: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        kernel = [
            (rule // (self.num_states ** k) % self.num_states)
            for k in range(self.num_states ** self.kernel_size)
        ]
        return torch.Tensor(kernel).to(dtype)


def ca1_replace_step(
        input: torch.Tensor,
        lookup: torch.Tensor,
        num_neighbours: int,
        num_states: int,
        iterations: int,
        wrap: bool = False,
) -> torch.Tensor:
    if input.ndim != 1:
        raise ValueError(f"Expected input.ndim=1, got {input.shape}")
    if lookup.ndim != 1:
        raise ValueError(f"Expected lookup.ndim=1, got {lookup.shape}")

    width = input.shape[-1]
    state = input.view(1, width)
    n_size = 1 + 2 * num_neighbours             # size of neighbourhood
    width_n = (width // n_size + 1) * n_size    # with rounded up to the next n_size multiple
    # multiplier to get the lookup index for each cell in local neighbourhood
    index_mult = torch.Tensor([2 ** n for n in range(n_size)]).view(1, 1, -1).to(state)

    history = [state]
    for it in range(iterations):

        padding = (1, n_size + width_n - state.shape[-1] - 1)
        if wrap:
            state = torch.concat([state[:, -padding[0]:], state, state[:, :padding[1]]], dim=-1)
        else:
            state = F.pad(state, padding)

        index = torch.empty(1, width_n, dtype=torch.int64)
        for k in range(n_size):
            state_slices = state[:, k: width_n + k].view(-1, width_n // n_size, n_size)
            #print(f"k={k}, state={state.shape}, width_n={width_n}, state_slices={state_slices.shape}")
            index_slices = (state_slices * index_mult).sum(dim=-1)
            #print("A", index.shape, index_slices.shape)
            index[:, k::n_size] = index_slices

        state = torch.index_select(lookup, 0, index[0])#.unsqueeze(0)
        state = state[None, :width]

        history.append(state)
    # print("H", [i.shape for i in history])
    return torch.concat(history)
