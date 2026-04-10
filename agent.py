"""
Dueling Double DQN agent with Prioritized Experience Replay (PER).

Architecture upgrades over vanilla DDQN:
  - Dueling Network: Splits Q(s,a) = V(s) + A(s,a) - mean(A) for better
    state evaluation even when actions don't matter much.
  - Prioritized Experience Replay: Samples high-TD-error transitions more
    frequently, accelerating learning on surprising outcomes.
  - Double DQN: Decouples action selection (main net) from evaluation
    (target net) to reduce overestimation bias.

Backward compatible: `DQNAgent.load()` auto-detects old model format
and loads into the legacy QNetwork architecture seamlessly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Q-networks
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Standard MLP Q-network (legacy architecture).
    Kept for backward compatibility with old saved models.
    """
    def __init__(self, obs_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (Wang et al., 2016).

    Splits the Q-value into two streams:
      Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))

    The Value stream learns "how good is this state?"
    The Advantage stream learns "how much better is action a vs. average?"

    This decomposition improves learning efficiency because the agent
    can learn the value of a state independently of action effects,
    which is especially useful when many actions have similar outcomes.
    """
    def __init__(self, obs_size: int, num_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
        )
        # Value stream: scalar state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream: per-action advantage A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)              # (batch, 1)
        advantage = self.advantage_stream(features)      # (batch, actions)
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    """Hyperparameters for Dueling DDQN + PER training."""
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    replay_size: int = 100_000
    min_replay_size: int = 2_000
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    epsilon_decay_mult: float = 0.998
    epsilon_reset_every_episodes: int = 0
    epsilon_reset_value: float = 0.3
    max_grad_norm: float = 1.0
    # PER hyperparameters
    per_alpha: float = 0.6      # prioritization exponent (0 = uniform, 1 = full priority)
    per_beta_start: float = 0.4 # importance sampling correction (anneals to 1.0)
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 100_000
    per_epsilon: float = 1e-6   # small constant to prevent zero priority


# ---------------------------------------------------------------------------
# Prioritized Experience Replay buffer
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for O(log N) prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = [None] * self.capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        return float(max(self.tree[leaf_start:leaf_start + self.size])) if self.size > 0 else 1.0

    def add(self, priority: float, data) -> None:
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016).

    Samples transitions with probability proportional to their TD-error,
    so the agent focuses learning on "surprising" transitions.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, seed: int = 0):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        data = (s.astype(np.float32), int(a), float(r), s2.astype(np.float32), bool(done))
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, data)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Sample a batch with importance-sampling weights."""
        indices = []
        priorities = []
        batch = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s_val = float(self.rng.uniform(low, high))
            idx, priority, data = self.tree.get(s_val)
            if data is None:
                # Fallback: resample from valid range
                s_val = float(self.rng.uniform(0, self.tree.total))
                idx, priority, data = self.tree.get(s_val)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        if len(batch) == 0:
            raise RuntimeError("PER buffer sampling failed — buffer may be empty")

        # Importance-sampling weights
        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / (self.tree.total + 1e-12)
        weights = (len(self) * probs + 1e-12) ** (-beta)
        weights = weights / (weights.max() + 1e-12)  # normalize

        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray, epsilon: float = 1e-6) -> None:
        for idx, td in zip(indices, td_errors):
            priority = (abs(float(td)) + epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)


# Legacy uniform replay buffer (kept for backward compat)
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.capacity
        )

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buf.append(
            (s.astype(np.float32), int(a), float(r), s2.astype(np.float32), bool(done))
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = self.rng.sample(self.buf, k=int(batch_size))
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Dueling Double DQN Agent with PER
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Production-grade Dueling Double DQN Agent with Prioritized Experience Replay.

    Key upgrades:
      1. Dueling Architecture: Q(s,a) = V(s) + A(s,a) - mean(A)
      2. Prioritized Replay: Focus learning on high-error transitions
      3. Double DQN: Decouple selection from evaluation
      4. Input Normalization: Min-Max scaling for stable gradients

    Backward compatible: loads old QNetwork models seamlessly.
    """

    NORM_DENOMS = np.array([12.0, 100.0, 30.0, 50.0, 50.0, 50.0, 200.0], dtype=np.float32)

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        config: Optional[DQNConfig] = None,
        seed: int = 0,
        device: Optional[str] = None,
        use_dueling: bool = True,
        use_per: bool = True,
    ):
        self.obs_size = int(obs_size)
        self.num_actions = int(num_actions)
        self.cfg = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        self.use_dueling = use_dueling
        self.use_per = use_per

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Networks — choose architecture
        NetClass = DuelingQNetwork if use_dueling else QNetwork
        self.q = NetClass(self.obs_size, self.num_actions).to(self.device)
        self.target = NetClass(self.obs_size, self.num_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=self.cfg.lr)

        # Replay buffer — choose type
        if use_per:
            self.replay = PrioritizedReplayBuffer(
                self.cfg.replay_size, alpha=self.cfg.per_alpha, seed=seed
            )
        else:
            self.replay = ReplayBuffer(self.cfg.replay_size, seed=seed)

        self.train_steps: int = 0
        self._epsilon_value: float = float(self.cfg.epsilon_start)
        self.episodes_seen: int = 0
        self._beta: float = float(self.cfg.per_beta_start)

    # --- Pipeline Steps ---

    def preprocess_state(self, obs: np.ndarray) -> torch.Tensor:
        """Normalize raw observation to [0, 1] range."""
        norm_obs = obs.astype(np.float32) / self.NORM_DENOMS
        return torch.tensor(norm_obs, dtype=torch.float32, device=self.device)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection on the main network."""
        if (not greedy) and (self.rng.random() < self.epsilon()):
            return int(self.rng.integers(0, self.num_actions))
        with torch.no_grad():
            q_values = self.predict_q_values(obs)
            return int(np.argmax(q_values))

    def predict_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return raw Q-values for XAI transparency."""
        with torch.no_grad():
            x = self.preprocess_state(obs).unsqueeze(0)
            q_values = self.q(x).squeeze(0)
            return q_values.cpu().numpy()

    # --- Training Logic ---

    def train_step(self) -> Dict[str, float]:
        """
        Single training update with Dueling DDQN + PER.
        """
        if not self.can_train():
            return {"loss": float("nan")}

        if self.use_per:
            # Anneal beta toward 1.0
            self._beta = min(
                self.cfg.per_beta_end,
                self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start)
                * self.train_steps / max(1, self.cfg.per_beta_anneal_steps)
            )
            s, a, r, s2, d, weights, indices = self.replay.sample(
                self.cfg.batch_size, beta=self._beta
            )
            w_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)
        else:
            s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
            w_t = torch.ones(self.cfg.batch_size, 1, device=self.device)
            indices = None

        # Preprocess
        s_t = self.preprocess_state(s)
        s2_t = self.preprocess_state(s2)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        d_t = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Current Q-values
        q_sa = self.q(s_t).gather(1, a_t)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.q(s2_t).argmax(dim=1, keepdim=True)
            q_target_next = self.target(s2_t).gather(1, next_actions)
            target_val = r_t + (1.0 - d_t) * self.cfg.gamma * q_target_next

        # TD errors for PER priority update
        td_errors = (q_sa - target_val).detach()

        # Weighted loss
        elementwise_loss = nn.functional.smooth_l1_loss(q_sa, target_val, reduction='none')
        loss = (w_t * elementwise_loss).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        # Update PER priorities
        if self.use_per and indices is not None:
            self.replay.update_priorities(
                indices,
                td_errors.squeeze(-1).cpu().numpy(),
                epsilon=self.cfg.per_epsilon,
            )

        # Housekeeping
        self.train_steps += 1
        self._epsilon_value = max(
            float(self.cfg.epsilon_end),
            float(self._epsilon_value) * float(self.cfg.epsilon_decay_mult),
        )
        if self.train_steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon()),
            "avg_q": float(q_sa.mean().item()),
        }

    # --- Helpers ---

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Legacy helper wrapping select_action."""
        return self.select_action(obs, greedy=greedy)

    def observe(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.replay.add(s, a, r, s2, done)

    def can_train(self) -> bool:
        return len(self.replay) >= self.cfg.min_replay_size

    def epsilon(self) -> float:
        return float(self._epsilon_value)

    def on_episode_end(self) -> None:
        self.episodes_seen += 1

    def save(self, path: str) -> None:
        payload = {
            "obs_size": self.obs_size,
            "num_actions": self.num_actions,
            "config": self.cfg.__dict__,
            "state_dict": self.q.state_dict(),
            "norm_denoms": self.NORM_DENOMS.tolist(),
            "architecture": "dueling" if self.use_dueling else "standard",
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)

        # Detect architecture from saved model
        arch = payload.get("architecture", "standard")  # old models = "standard"
        use_dueling = (arch == "dueling")

        # Filter out PER-specific keys that old configs won't have
        config_dict = {}
        valid_fields = {f.name for f in DQNConfig.__dataclass_fields__.values()}
        for k, v in payload.get("config", {}).items():
            if k in valid_fields:
                config_dict[k] = v

        cfg = DQNConfig(**config_dict)
        agent = cls(
            payload["obs_size"],
            payload["num_actions"],
            cfg,
            seed=0,
            device=device,
            use_dueling=use_dueling,
            use_per=False,  # Don't need PER for inference
        )
        agent.q.load_state_dict(payload["state_dict"])
        agent.target.load_state_dict(payload["state_dict"])
        agent.target.eval()
        return agent
