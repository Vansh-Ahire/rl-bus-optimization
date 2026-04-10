"""
OpenEnv-compliant RL environment for bus route optimisation.

This module keeps **all** original MiniBusEnv logic intact and wraps it with
Pydantic-typed interfaces required by the OpenEnv specification:

    Observation, Action, Reward — typed models
    reset()  -> Observation
    step()   -> (Observation, Reward, done, info)
    state()  -> dict
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

# Optional GTFS demand profile integration
try:
    from data.gtfs_profiles import DemandProfile, get_demand_profile
except ImportError:
    DemandProfile = None  # type: ignore
    get_demand_profile = None  # type: ignore


# ---------------------------------------------------------------------------
# Pydantic models (OpenEnv interface)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Structured observation returned by the environment."""

    bus_position: int = Field(..., description="Current stop index of the controlled bus")
    fuel: float = Field(..., description="Remaining fuel (0-100)")
    onboard_passengers: int = Field(..., description="Number of passengers currently on board")
    queue_current_stop: int = Field(..., description="Queue length at the current stop")
    queue_next_stop: int = Field(..., description="Queue length at the next stop")
    queue_next_next_stop: int = Field(..., description="Queue length at the stop after next")
    time_step: int = Field(..., description="Current simulation time step")

    def to_array(self) -> np.ndarray:
        """Convert to the flat float32 array expected by neural-net agents."""
        return np.array(
            [
                float(self.bus_position),
                float(self.fuel),
                float(self.onboard_passengers),
                float(self.queue_current_stop),
                float(self.queue_next_stop),
                float(self.queue_next_next_stop),
                float(self.time_step),
            ],
            dtype=np.float32,
        )

    class Config:
        arbitrary_types_allowed = True


class Action(BaseModel):
    """Discrete action taken by the agent."""

    action: int = Field(
        ...,
        ge=0,
        le=2,
        description="0 = move+pickup, 1 = move+skip, 2 = wait+pickup",
    )


class Reward(BaseModel):
    """Scalar reward with an optional breakdown."""

    value: float = Field(..., description="Scalar reward for the step")
    passengers_picked: int = Field(0, description="Passengers picked up this step")
    fuel_used: float = Field(0.0, description="Fuel consumed this step")
    penalties_applied: List[str] = Field(
        default_factory=list,
        description="Human-readable list of penalty/bonus tags applied",
    )


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from the original project)
# ---------------------------------------------------------------------------

@dataclass
class StepStats:
    passengers_picked: int = 0
    picked_wait_times: Optional[np.ndarray] = None
    fuel_used: float = 0.0
    ignored_large_queue: bool = False


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class BusRoutingEnv:
    """
    OpenEnv-compliant RL environment for a simplified circular bus route.

    Keeps **all** original MiniBusEnv logic while exposing typed Pydantic
    interfaces (``Observation``, ``Action``, ``Reward``) and a ``state()``
    method as required by the OpenEnv spec.

    Action space (discrete, 3 actions):
        0 — move to next stop and pick up passengers
        1 — move to next stop but skip pickup
        2 — wait at current stop and pick up passengers

    Observation vector (7-d float32):
        [bus_stop_idx, fuel_0_100, onboard_passengers,
         queue_len_at_{pos, pos+1, pos+2}, time_step]
    """

    # Action constants ---
    ACTION_MOVE_PICKUP = 0
    ACTION_MOVE_SKIP = 1
    ACTION_WAIT = 2

    def __init__(
        self,
        num_stops: int = 10,
        num_buses: int = 1,
        max_steps: int = 150,
        seed: int = 0,
        bus_capacity: int = 30,
        fuel_start: float = 100.0,
        passenger_arrival_rate: float = 1.2,
        large_queue_threshold: int = 10,
        wait_time_threshold: int = 3,
        fuel_cost_move: float = 1.0,
        fuel_cost_wait: float = 0.2,
        background_bus_pickup_fraction: float = 0.6,
        new_stop_bonus: float = 1.0,
        idle_camping_penalty: float = 0.6,
        camping_grace_steps: int = 1,
        nearby_queue_ignore_penalty: float = 1.5,
        recent_window: int = 10,
        recent_unvisited_bonus: float = 1.0,
        repeat_stop_penalty: float = 0.5,
        high_queue_reward_threshold: int = 6,
        high_queue_visit_bonus: float = 2.0,
        reward_clip: float = 10.0,
        demand_profile: str = "synthetic",
    ):
        # Support large-scale tasks up to 50 stops for hackathon evaluation
        if not (5 <= num_stops <= 50):
            raise ValueError("num_stops must be in [5, 50].")
        if not (1 <= num_buses <= 3):
            raise ValueError("num_buses must be in [1, 3].")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0.")

        self.num_stops = int(num_stops)
        self.num_buses = int(num_buses)
        self.max_steps = int(max_steps)
        self.bus_capacity = int(bus_capacity)
        self.fuel_start = float(fuel_start)
        self.passenger_arrival_rate = float(passenger_arrival_rate)
        self.large_queue_threshold = int(large_queue_threshold)
        self.wait_time_threshold = int(wait_time_threshold)
        self.fuel_cost_move = float(fuel_cost_move)
        self.fuel_cost_wait = float(fuel_cost_wait)
        self.background_bus_pickup_fraction = float(background_bus_pickup_fraction)
        self.new_stop_bonus = float(new_stop_bonus)
        self.idle_camping_penalty = float(idle_camping_penalty)
        self.camping_grace_steps = int(camping_grace_steps)
        self.nearby_queue_ignore_penalty = float(nearby_queue_ignore_penalty)
        self.recent_window = int(recent_window)
        self.recent_unvisited_bonus = float(recent_unvisited_bonus)
        self.repeat_stop_penalty = float(repeat_stop_penalty)
        self.high_queue_reward_threshold = int(high_queue_reward_threshold)
        self.high_queue_visit_bonus = float(high_queue_visit_bonus)
        self.reward_clip = float(reward_clip)

        # GTFS demand profile integration
        self.demand_profile_name = demand_profile
        self._demand_profile = None
        if demand_profile != "synthetic" and get_demand_profile is not None:
            try:
                self._demand_profile = get_demand_profile(demand_profile, num_stops)
            except Exception:
                self._demand_profile = None  # fallback to synthetic

        self.rng = np.random.default_rng(seed)

        # Mutable episode state
        self.t: int = 0
        self.bus_pos: int = 0
        self.fuel: float = self.fuel_start
        self.onboard: int = 0
        self.stop_queues: List[List[int]] = [[] for _ in range(self.num_stops)]
        self.visited_stops: set[int] = set()
        self.visit_counts: np.ndarray = np.zeros(self.num_stops, dtype=np.int32)
        self.recent_stops: Deque[int] = deque(maxlen=self.recent_window)
        self._consecutive_same_stop_steps: int = 0
        self._prev_pos: int = 0

        # Metrics
        self.total_picked: int = 0
        self.total_wait_time_picked: float = 0.0
        self.total_fuel_used: float = 0.0
        self.total_reward: float = 0.0

        # Background buses
        self.bg_bus_pos: List[int] = [0 for _ in range(max(0, self.num_buses - 1))]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_size(self) -> int:
        return 7

    @property
    def num_actions(self) -> int:
        return 3

    # ------------------------------------------------------------------
    # OpenEnv — state()
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the full environment state."""
        return {
            "t": self.t,
            "bus_pos": self.bus_pos,
            "fuel": self.fuel,
            "onboard": self.onboard,
            "stop_queues": [list(q) for q in self.stop_queues],
            "visited_stops": sorted(self.visited_stops),
            "visit_counts": self.visit_counts.tolist(),
            "recent_stops": list(self.recent_stops),
            "consecutive_same_stop_steps": self._consecutive_same_stop_steps,
            "total_picked": self.total_picked,
            "total_wait_time_picked": self.total_wait_time_picked,
            "total_fuel_used": self.total_fuel_used,
            "total_reward": self.total_reward,
            "bg_bus_pos": list(self.bg_bus_pos),
            "num_stops": self.num_stops,
            "max_steps": self.max_steps,
        }

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # OpenEnv — reset()
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self.t = 0
        self.bus_pos = int(self.rng.integers(0, self.num_stops))
        self._prev_pos = self.bus_pos
        self.fuel = float(self.fuel_start)
        self.onboard = 0
        self.stop_queues = [[] for _ in range(self.num_stops)]
        self.visited_stops = {self.bus_pos}
        self.visit_counts = np.zeros(self.num_stops, dtype=np.int32)
        self.visit_counts[self.bus_pos] += 1
        self.recent_stops = deque([self.bus_pos], maxlen=self.recent_window)
        self._consecutive_same_stop_steps = 0

        self.total_picked = 0
        self.total_wait_time_picked = 0.0
        self.total_fuel_used = 0.0
        self.total_reward = 0.0

        self.bg_bus_pos = [
            int(self.rng.integers(0, self.num_stops))
            for _ in range(max(0, self.num_buses - 1))
        ]
        return self._make_observation()

    # ------------------------------------------------------------------
    # Internal helpers (untouched logic from the original project)
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        q0 = len(self.stop_queues[self.bus_pos])
        q1 = len(self.stop_queues[(self.bus_pos + 1) % self.num_stops])
        q2 = len(self.stop_queues[(self.bus_pos + 2) % self.num_stops])
        return Observation(
            bus_position=self.bus_pos,
            fuel=self.fuel,
            onboard_passengers=self.onboard,
            queue_current_stop=q0,
            queue_next_stop=q1,
            queue_next_next_stop=q2,
            time_step=self.t,
        )

    def render(self) -> Dict[str, Any]:
        """
        Return a visual representation of the current route state.
        Used by the UI to show stop queues and bus location.
        """
        return {
            "bus_pos": self.bus_pos,
            "stops": [
                {
                    "stop_idx": i,
                    "queue_len": len(self.stop_queues[i]),
                    "is_bus_here": (i == self.bus_pos),
                }
                for i in range(self.num_stops)
            ],
            "fuel": float(self.fuel),
            "onboard": int(self.onboard),
            "total_reward": float(self.total_reward),
            "time_step": int(self.t),
        }

    def _get_obs(self) -> np.ndarray:
        """Legacy helper — returns raw float32 array for backward compat."""
        return self._make_observation().to_array()

    def _increment_waits(self) -> None:
        for s in range(self.num_stops):
            if self.stop_queues[s]:
                self.stop_queues[s] = [w + 1 for w in self.stop_queues[s]]

    def _arrive_passengers(self) -> None:
        if self._demand_profile is not None:
            # GTFS-calibrated: per-stop, time-varying arrival rates
            for s in range(self.num_stops):
                rate = self._demand_profile.get_arrival_rate(
                    self.passenger_arrival_rate, s, self.t
                )
                k = int(self.rng.poisson(max(0.01, rate)))
                if k > 0:
                    self.stop_queues[s].extend([0] * k)
        else:
            # Legacy synthetic: uniform Poisson across all stops
            arrivals = self.rng.poisson(self.passenger_arrival_rate, size=self.num_stops)
            for s, k in enumerate(arrivals.tolist()):
                if k > 0:
                    self.stop_queues[s].extend([0] * int(k))

    def _pickup_at_stop(
        self, stop_idx: int, capacity_left: int
    ) -> Tuple[int, np.ndarray]:
        q = self.stop_queues[stop_idx]
        if not q or capacity_left <= 0:
            return 0, np.array([], dtype=np.float32)
        k = min(len(q), int(capacity_left))
        picked = np.array(q[:k], dtype=np.float32)
        self.stop_queues[stop_idx] = q[k:]
        return int(k), picked

    def _step_background_buses(self) -> None:
        for i in range(len(self.bg_bus_pos)):
            pos = (self.bg_bus_pos[i] + 1) % self.num_stops
            self.bg_bus_pos[i] = pos
            q = self.stop_queues[pos]
            if not q:
                continue
            take = int(np.floor(len(q) * self.background_bus_pickup_fraction))
            if take <= 0:
                continue
            self.stop_queues[pos] = q[take:]

    # ------------------------------------------------------------------
    # OpenEnv — step()
    # ------------------------------------------------------------------

    def step(
        self, action: Action | int
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Accepts either an ``Action`` model or a plain int for backward
        compatibility with existing training code.
        """
        if isinstance(action, Action):
            act = action.action
        else:
            act = int(action)

        if act not in (0, 1, 2):
            raise ValueError(
                "Invalid action. Must be 0 (move+pickup), 1 (move+skip), 2 (wait)."
            )

        # --- passenger dynamics ---
        self._increment_waits()
        self._arrive_passengers()
        self._step_background_buses()

        stats = StepStats()
        reward = 0.0
        visited_new_stop = False
        moved = act in (self.ACTION_MOVE_PICKUP, self.ACTION_MOVE_SKIP)
        penalty_tags: List[str] = []

        current_stop = self.bus_pos
        next_stop = (self.bus_pos + 1) % self.num_stops
        next_stop_queue_len_before = len(self.stop_queues[next_stop])

        # --- apply action ---
        if act == self.ACTION_WAIT:
            fuel_used = self.fuel_cost_wait
            self.fuel -= fuel_used
            stats.fuel_used = fuel_used
            capacity_left = self.bus_capacity - self.onboard
            picked_n, picked_waits = self._pickup_at_stop(self.bus_pos, capacity_left)
            self.onboard += picked_n
            stats.passengers_picked = picked_n
            stats.picked_wait_times = picked_waits
        else:
            fuel_used = self.fuel_cost_move
            self.fuel -= fuel_used
            stats.fuel_used = fuel_used
            self.bus_pos = (self.bus_pos + 1) % self.num_stops
            if self.bus_pos not in self.visited_stops:
                visited_new_stop = True
            self.visited_stops.add(self.bus_pos)
            self.visit_counts[self.bus_pos] += 1

            if act == self.ACTION_MOVE_PICKUP:
                capacity_left = self.bus_capacity - self.onboard
                picked_n, picked_waits = self._pickup_at_stop(
                    self.bus_pos, capacity_left
                )
                self.onboard += picked_n
                stats.passengers_picked = picked_n
                stats.picked_wait_times = picked_waits
            else:
                stats.passengers_picked = 0
                stats.picked_wait_times = np.array([], dtype=np.float32)

        # --- reward shaping ---
        reward += 2.0 * stats.passengers_picked
        if stats.passengers_picked > 0:
            penalty_tags.append(f"+pickup({stats.passengers_picked})")

        if (
            stats.picked_wait_times is not None
            and stats.picked_wait_times.size > 0
        ):
            if float(stats.picked_wait_times.mean()) <= float(
                self.wait_time_threshold
            ):
                reward += 5.0
                penalty_tags.append("+low_wait_bonus")

        reward -= 1.0 * float(stats.fuel_used)
        penalty_tags.append(f"-fuel({stats.fuel_used:.1f})")

        if act == self.ACTION_MOVE_SKIP:
            ignored_stop = self.bus_pos
            if len(self.stop_queues[ignored_stop]) >= self.large_queue_threshold:
                reward -= 3.0
                stats.ignored_large_queue = True
                penalty_tags.append("-ignored_large_queue")

        if act == self.ACTION_WAIT:
            q1 = len(self.stop_queues[(self.bus_pos + 1) % self.num_stops])
            q2 = len(self.stop_queues[(self.bus_pos + 2) % self.num_stops])
            if max(q1, q2) >= self.large_queue_threshold:
                reward -= self.nearby_queue_ignore_penalty
                penalty_tags.append("-nearby_queue_ignored")

        done = False
        if self.fuel <= 0.0:
            reward -= 10.0
            done = True
            penalty_tags.append("-fuel_depleted")

        if visited_new_stop:
            reward += self.new_stop_bonus
            penalty_tags.append("+new_stop")

        if moved and (next_stop not in self.recent_stops):
            reward += self.recent_unvisited_bonus
            penalty_tags.append("+unvisited_recently")

        if self.bus_pos == current_stop and act == self.ACTION_WAIT:
            reward -= self.repeat_stop_penalty
            penalty_tags.append("-repeat_stop")

        if moved and next_stop_queue_len_before >= self.high_queue_reward_threshold:
            reward += self.high_queue_visit_bonus
            penalty_tags.append("+high_demand_visit")

        if self.bus_pos == self._prev_pos:
            self._consecutive_same_stop_steps += 1
        else:
            self._consecutive_same_stop_steps = 0
        if self._consecutive_same_stop_steps > self.camping_grace_steps:
            reward -= self.idle_camping_penalty
            penalty_tags.append("-idle_camping")
        self._prev_pos = self.bus_pos

        self.recent_stops.append(self.bus_pos)

        if self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.t += 1
        if self.t >= self.max_steps:
            done = True

        # --- metrics ---
        self.total_reward += float(reward)
        self.total_fuel_used += float(stats.fuel_used)
        self.total_picked += int(stats.passengers_picked)
        if (
            stats.picked_wait_times is not None
            and stats.picked_wait_times.size > 0
        ):
            self.total_wait_time_picked += float(stats.picked_wait_times.sum())

        info: Dict[str, Any] = {
            "t": self.t,
            "bus_pos": self.bus_pos,
            "fuel": self.fuel,
            "onboard": self.onboard,
            "step_passengers_picked": stats.passengers_picked,
            "step_mean_wait_picked": (
                float(stats.picked_wait_times.mean())
                if stats.picked_wait_times is not None
                and stats.picked_wait_times.size > 0
                else None
            ),
            "step_fuel_used": float(stats.fuel_used),
            "ignored_large_queue": bool(stats.ignored_large_queue),
            "visited_new_stop": bool(visited_new_stop),
            "consecutive_same_stop_steps": int(self._consecutive_same_stop_steps),
            "episode_total_reward": float(self.total_reward),
            "episode_total_picked": int(self.total_picked),
            "episode_total_fuel_used": float(self.total_fuel_used),
            "episode_avg_wait_picked": (
                self.total_wait_time_picked / self.total_picked
            )
            if self.total_picked > 0
            else None,
            "stop_coverage": float(len(self.visited_stops) / self.num_stops),
        }

        reward_model = Reward(
            value=float(reward),
            passengers_picked=int(stats.passengers_picked),
            fuel_used=float(stats.fuel_used),
            penalties_applied=penalty_tags,
        )

        return self._make_observation(), reward_model, bool(done), info

    # ------------------------------------------------------------------
    # Utility: run a full episode (backward-compatible)
    # ------------------------------------------------------------------

    def run_episode(
        self,
        policy_fn,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a single episode with *policy_fn(obs_array) -> int* and return
        aggregate metrics.  This preserves backward compatibility with the
        existing training / grading code.
        """
        obs_model = self.reset()
        obs = obs_model.to_array()
        done = False
        steps = 0
        while not done:
            action = int(policy_fn(obs))
            obs_model, reward_model, done, _info = self.step(action)
            obs = obs_model.to_array()
            steps += 1
            if max_steps is not None and steps >= int(max_steps):
                break

        avg_wait = (
            (self.total_wait_time_picked / self.total_picked)
            if self.total_picked > 0
            else float("inf")
        )
        counts = self.visit_counts.astype(np.float64)
        if counts.sum() > 0:
            p = counts / counts.sum()
            entropy = float(-(p[p > 0] * np.log(p[p > 0] + 1e-12)).sum())
            max_entropy = float(np.log(self.num_stops))
            route_entropy = float(entropy / (max_entropy + 1e-12))
            max_stop_fraction = float(p.max())
        else:
            route_entropy = 0.0
            max_stop_fraction = 1.0

        return {
            "total_reward": float(self.total_reward),
            "avg_wait_time": float(avg_wait),
            "fuel_used": float(self.total_fuel_used),
            "stop_coverage": float(len(self.visited_stops) / self.num_stops),
            "route_entropy": float(route_entropy),
            "max_stop_fraction": float(max_stop_fraction),
            "passengers_picked": float(self.total_picked),
            "steps": float(steps),
        }


# Backward-compatible alias so old imports still work
MiniBusEnv = BusRoutingEnv
