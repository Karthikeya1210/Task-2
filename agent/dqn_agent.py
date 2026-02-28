"""
DQN Agent for Chef's Hat Card Game
Module: 7043SCN — Generative AI and Reinforcement Learning
Variant: ID mod 7 = 5 → Robustness & Generalisation

Chef's Hat GYM v3 uses an async API:
    pip install chefshatgym

The agent extends the base ChefsHat agent interface and implements
get_action() to select moves using a trained DQN policy.
"""

import os
import copy
import json
import random
import logging
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────── CONFIG ──────────────────────────────────────────

DQN_CONFIG = {
    "hidden_sizes":   [256, 256],
    "dropout":        0.2,
    "lr":             1e-4,
    "gamma":          0.99,
    "batch_size":     64,
    "buffer_size":    50_000,
    "target_update":  500,
    "min_buffer":     1_000,
    "episodes":       3_000,
    "max_steps":      500,
    "eps_start":      1.0,
    "eps_end":        0.05,
    "eps_decay":      0.995,
    "checkpoint_dir": "./outputs/checkpoints",
}

# Chef's Hat observation size (flattened)
# Hand: 200 cards one-hot + pile: 200 + scores: 4 + phase: 1 = 405
# We use a fixed 405-dim state from the observation array
STATE_DIM  = 405
ACTION_DIM = 200   # max possible actions in Chef's Hat


# ─────────────────────────── NETWORK ─────────────────────────────────────────

class DuelingDQN(nn.Module):
    """Dueling DQN: separates V(s) and A(s,a) for better credit assignment."""

    def __init__(self, state_dim: int, action_dim: int, hidden: list, dropout: float):
        super().__init__()
        layers, in_d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        self.shared = nn.Sequential(*layers)
        self.value     = nn.Sequential(nn.Linear(in_d, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(in_d, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.shared(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))  # Dueling aggregation


# ─────────────────────────── REPLAY BUFFER ───────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((np.array(s, dtype=np.float32), int(a), float(r),
                         np.array(s2, dtype=np.float32), bool(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.stack(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.stack(s2)),
                torch.FloatTensor(d))

    def __len__(self): return len(self.buf)


# ─────────────────────────── DQN CORE ────────────────────────────────────────

class DQNCore:
    """The learning/inference engine, separate from the game interface."""

    def __init__(self, state_dim, action_dim, config=None):
        self.cfg    = config or DQN_CONFIG
        self.device = torch.device("cpu")
        self.action_dim = action_dim

        self.policy = DuelingDQN(state_dim, action_dim,
                                  self.cfg["hidden_sizes"], self.cfg["dropout"]).to(self.device)
        self.target = copy.deepcopy(self.policy)
        self.target.eval()

        self.opt     = optim.Adam(self.policy.parameters(), lr=self.cfg["lr"], eps=1e-5)
        self.memory  = ReplayBuffer(self.cfg["buffer_size"])
        self.epsilon = self.cfg["eps_start"]
        self.steps   = 0
        self.updates = 0
        self.losses  = []

    def act(self, state: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        self.epsilon = max(self.cfg["eps_end"], self.epsilon * self.cfg["eps_decay"])
        if random.random() < self.epsilon:
            return random.choice(valid_actions) if valid_actions else random.randrange(self.action_dim)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(s).squeeze(0)
        if valid_actions is not None:
            mask = torch.full((self.action_dim,), float("-inf"))
            mask[valid_actions] = 0.0
            q = q + mask.to(self.device)
        return int(q.argmax().item())

    def learn(self):
        if len(self.memory) < self.cfg["min_buffer"]: return None
        s, a, r, s2, d = self.memory.sample(self.cfg["batch_size"])
        s, a, r, s2, d = [x.to(self.device) for x in [s, a, r, s2, d]]

        curr_q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.policy(s2).argmax(1)
            next_q = self.target(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target_q = r + self.cfg["gamma"] * next_q * (1 - d)

        loss = nn.SmoothL1Loss()(curr_q, target_q)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.cfg["target_update"] == 0:
            self.target.load_state_dict(self.policy.state_dict())

        lv = loss.item()
        self.losses.append(lv)
        return lv

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"policy": self.policy.state_dict(),
                    "target": self.target.state_dict(),
                    "opt":    self.opt.state_dict(),
                    "eps":    self.epsilon,
                    "steps":  self.steps}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.target.load_state_dict(ck["target"])
        self.opt.load_state_dict(ck["opt"])
        self.epsilon = ck["eps"]


# ─────────────────────────── STATE ENCODING ──────────────────────────────────

def encode_observation(obs) -> np.ndarray:
    """
    Convert Chef's Hat GYM v3 observation to flat numpy vector.
    The observation from ChefsHatGym is already a numpy array —
    we normalise and pad/truncate to STATE_DIM.
    """
    if obs is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    arr = np.array(obs, dtype=np.float32).flatten()
    if len(arr) >= STATE_DIM:
        return arr[:STATE_DIM]
    return np.pad(arr, (0, STATE_DIM - len(arr)))


def shape_reward(raw_reward: float, prev_hand: int, curr_hand: int) -> float:
    """
    Augment sparse end-of-game reward with dense shaping signals.
    Raw: +1 win, -1 lose (end of match only — very sparse).
    Shaped: card-shedding bonus, time penalty.
    """
    shaped  = raw_reward
    shaped += 0.05 * max(0, prev_hand - curr_hand)   # reward card reduction
    shaped -= 0.01                                     # step penalty
    return float(np.clip(shaped, -2.0, 2.0))


# ─────────────────────────── GYM V3 AGENT WRAPPER ────────────────────────────

try:
    from agents.base_agent import BaseAgent  # ChefsHatGYM v3

    class DQNAgent(BaseAgent):
        """
        DQN agent conforming to Chef's Hat GYM v3 interface.
        Extends BaseAgent and implements get_action().
        """

        def __init__(self, name: str, config=None, train: bool = True):
            super().__init__(name=name)
            self.dqn   = DQNCore(STATE_DIM, ACTION_DIM, config or DQN_CONFIG)
            self.train = train
            self._prev_obs  = None
            self._prev_action = None
            self._prev_hand   = 0

        def get_exhanged_cards(self, cards, number_of_cards):
            """Required by BaseAgent for card exchange phase."""
            return cards[:number_of_cards]

        def get_action(self, observation, action_mask=None):
            """
            Called by the Room on each turn.
            observation: numpy array from the environment
            action_mask: list of valid action indices
            """
            state = encode_observation(observation)
            curr_hand = int(np.sum(observation[:200]) if observation is not None else 0)

            # Learn from previous transition
            if self.train and self._prev_obs is not None:
                reward = shape_reward(0.0, self._prev_hand, curr_hand)
                self.dqn.memory.push(self._prev_obs, self._prev_action, reward, state, False)
                self.dqn.learn()

            # Get valid actions from mask
            if action_mask is not None:
                valid = [i for i, v in enumerate(action_mask) if v == 1]
            else:
                valid = None

            action = self.dqn.act(state, valid)

            self._prev_obs    = state
            self._prev_action = action
            self._prev_hand   = curr_hand
            return action

        def update_end_game(self, observation, reward, info=None):
            """Called at end of match with terminal reward."""
            if self.train and self._prev_obs is not None:
                state = encode_observation(observation)
                shaped = shape_reward(reward, self._prev_hand, 0)
                self.dqn.memory.push(self._prev_obs, self._prev_action, shaped, state, True)
                self.dqn.learn()
            self._prev_obs    = None
            self._prev_action = None

        def save(self, path): self.dqn.save(path)
        def load(self, path): self.dqn.load(path)

except ImportError:
    # chefshatgym not installed — define a stub for simulate mode
    class DQNAgent:
        def __init__(self, name, config=None, train=True):
            self.name = name
            self.dqn  = DQNCore(STATE_DIM, ACTION_DIM, config or DQN_CONFIG)
        def save(self, path): self.dqn.save(path)
        def load(self, path): self.dqn.load(path)


if __name__ == "__main__":
    print("DQN Agent module loaded.")
    print(f"State dim: {STATE_DIM} | Action dim: {ACTION_DIM}")
    core = DQNCore(STATE_DIM, ACTION_DIM)
    test_state = np.random.randn(STATE_DIM).astype(np.float32)
    action = core.act(test_state)
    print(f"Test action selected: {action}")
    print("OK — run training/train.py to train the agent.")