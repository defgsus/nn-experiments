import time
from typing import List, Tuple

import random
import math

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from .bdenv import BoulderDashEnvironment
from src.util import *


class BoulderDashRLTrainer:

    def __init__(
            self,
            environment: BoulderDashEnvironment,
            policy_model: nn.Module,
            target_model: nn.Module,
            num_episodes: int = 100,
            max_trials: int = 100,
            batch_size: int = 4,
            memory_size: int = 10000,
            discount_factor: float = 0.999,
            device: str = "auto",
    ):
        self.device = to_torch_device(device)
        self.environment = environment
        self.policy_model = policy_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.num_episodes = num_episodes
        self.max_trials = max_trials
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.discount_factor = discount_factor
        self._memory_idx = 0
        self.current_episode = 0
        self.optimizer = optim.RMSprop(self.policy_model.parameters())
        self.memory: List[Tuple] = []

    def train(self):
        for i in tqdm(range(self.num_episodes), desc="episode"):
            self._train_episode()
            self.eval()

    def eval(self, seed=23, console: bool = False):
        self.environment.reset(seed=seed)

        rewards = []
        for idx in tqdm(range(100), desc="evaluation", disable=console):
            state = self.environment.state().to(self.device).unsqueeze(0)
            action = self.target_model.forward(state).argmax().item()
            result, reward, terminated = self.environment.act(action)
            rewards.append(reward)

            if console:
                print(f"\n-- step {idx} --")
                self.environment.bd.dump(ansi_colors=True)
                print("action:", self.environment.bd.ACTIONS.get(action))

                print("reward:", reward, self.environment.bd.RESULTS.get(result), "(terminated)" if terminated else "")
                time.sleep(.2)

            if terminated:
                break

        print(rewards)

    def _train_episode(self):
        self.environment.reset()
        state = self.environment.state().to(self.device).unsqueeze(0)

        for t in range(self.max_trials):
            action = self._get_action(state)

            _, reward, terminated = self.environment.act(action)
            reward = torch.tensor([reward], device=self.device)

            next_state = None
            if not terminated:
                next_state = self.environment.state().to(self.device).unsqueeze(0)

            self._push_memory((state, action, next_state, reward))

            if len(self.memory) >= self.batch_size:
                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                self._model_update()

            if terminated:
                break

            state = next_state

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self._sample_memory(self.batch_size)
        batch = [*zip(*transitions)]
        #print("BATCH", len(batch))
        #for b in batch:
        #    print(len(b))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch[2])),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch[2] if s is not None])
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[3])

        #print("state_batch ", state_batch.shape)
        #print("action_batch", action_batch.shape)
        #print("reward_batch", reward_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self._log_loss(float(loss))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

    def _push_memory(self, arg):
        if len(self.memory) < self.memory_size:
            self.memory.append(arg)
        else:
            self.memory[self._memory_idx % len(self.memory)] = arg
            self._memory_idx += 1

    def _sample_memory(self, count: int):
        return random.sample(self.memory, count)

    def _model_update(self, tau: float = 0.005):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1. - tau)
        self.target_model.load_state_dict(target_net_state_dict)

    def _get_action(self, map: torch.Tensor):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.current_episode / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                if map.ndim == 3:
                    map = map.unsqueeze(0)
                return self.policy_model(map).argmax(-1).view(1, 1)
        else:
            return torch.tensor(
                [[self.environment.get_random_action()]],
                device=self.device, dtype=torch.long,
            )

    def _log_loss(self, loss: float):
        pass
