"""
Multi-task configuration for the OpenEnv bus routing environment.

Three difficulty tiers — Easy, Medium, Hard — share the same
``BusRoutingEnv`` class but differ in the number of stops, passenger
demand, fuel constraints, and penalty intensity.

Now expanded to include 10 subtasks for each difficulty level (30 tasks total).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

from environment import BusRoutingEnv

# Explicitly export task configurations for OpenEnv detection
__all__ = [
    "TaskConfig",
    "TASKS",
    "TASK_EASY",
    "TASK_MEDIUM", 
    "TASK_HARD",
    "get_task",
]


@dataclass
class TaskConfig:
    """All parameters needed to instantiate a BusRoutingEnv for a task."""

    name: str = ""
    description: str = ""
    difficulty: str = "medium"  # easy | medium | hard

    num_stops: int = 10
    num_buses: int = 1
    max_steps: int = 150
    seed: int = 42
    bus_capacity: int = 30
    fuel_start: float = 100.0
    passenger_arrival_rate: float = 1.2
    large_queue_threshold: int = 10
    wait_time_threshold: int = 3
    fuel_cost_move: float = 1.0
    fuel_cost_wait: float = 0.2
    background_bus_pickup_fraction: float = 0.6

    new_stop_bonus: float = 1.0
    idle_camping_penalty: float = 0.6
    camping_grace_steps: int = 1
    nearby_queue_ignore_penalty: float = 1.5
    recent_window: int = 10
    recent_unvisited_bonus: float = 1.0
    repeat_stop_penalty: float = 0.5
    high_queue_reward_threshold: int = 6
    high_queue_visit_bonus: float = 2.0
    reward_clip: float = 10.0

    demand_profile: str = "synthetic"

    def build_env(self) -> BusRoutingEnv:
        import os
        m_steps = int(os.getenv("EVAL_MAX_STEPS", self.max_steps))
        return BusRoutingEnv(
            num_stops=self.num_stops,
            num_buses=self.num_buses,
            max_steps=m_steps,
            seed=self.seed,
            bus_capacity=self.bus_capacity,
            fuel_start=self.fuel_start,
            passenger_arrival_rate=self.passenger_arrival_rate,
            large_queue_threshold=self.large_queue_threshold,
            wait_time_threshold=self.wait_time_threshold,
            fuel_cost_move=self.fuel_cost_move,
            fuel_cost_wait=self.fuel_cost_wait,
            background_bus_pickup_fraction=self.background_bus_pickup_fraction,
            new_stop_bonus=self.new_stop_bonus,
            idle_camping_penalty=self.idle_camping_penalty,
            camping_grace_steps=self.camping_grace_steps,
            nearby_queue_ignore_penalty=self.nearby_queue_ignore_penalty,
            recent_window=self.recent_window,
            recent_unvisited_bonus=self.recent_unvisited_bonus,
            repeat_stop_penalty=self.repeat_stop_penalty,
            high_queue_reward_threshold=self.high_queue_reward_threshold,
            high_queue_visit_bonus=self.high_queue_visit_bonus,
            reward_clip=self.reward_clip,
            demand_profile=self.demand_profile,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "num_stops": self.num_stops,
            "num_buses": self.num_buses,
            "max_steps": self.max_steps,
            "fuel_start": self.fuel_start,
            "passenger_arrival_rate": self.passenger_arrival_rate,
            "fuel_cost_move": self.fuel_cost_move,
            "fuel_cost_wait": self.fuel_cost_wait,
            "large_queue_threshold": self.large_queue_threshold,
            "bus_capacity": self.bus_capacity,
        }


_TASK_EASY_TEMPLATE = TaskConfig(
    difficulty="easy",
    num_stops=5,
    num_buses=1,
    max_steps=100,
    seed=42,
    bus_capacity=30,
    fuel_start=100.0,
    passenger_arrival_rate=0.6,
    large_queue_threshold=12,
    wait_time_threshold=5,
    fuel_cost_move=0.5,
    fuel_cost_wait=0.1,
    new_stop_bonus=0.5,
    idle_camping_penalty=0.3,
    nearby_queue_ignore_penalty=0.5,
    repeat_stop_penalty=0.2,
    high_queue_reward_threshold=8,
    reward_clip=10.0,
    demand_profile="off_peak",
)

_TASK_MEDIUM_TEMPLATE = TaskConfig(
    difficulty="medium",
    num_stops=10,
    num_buses=1,
    max_steps=150,
    seed=42,
    bus_capacity=30,
    fuel_start=100.0,
    passenger_arrival_rate=1.2,
    large_queue_threshold=10,
    wait_time_threshold=3,
    fuel_cost_move=1.0,
    fuel_cost_wait=0.2,
    new_stop_bonus=1.0,
    idle_camping_penalty=0.6,
    nearby_queue_ignore_penalty=1.5,
    repeat_stop_penalty=0.5,
    high_queue_reward_threshold=6,
    reward_clip=10.0,
    demand_profile="weekday",
)

_TASK_HARD_TEMPLATE = TaskConfig(
    difficulty="hard",
    num_stops=12,
    num_buses=2,
    max_steps=200,
    seed=42,
    bus_capacity=25,
    fuel_start=80.0,
    passenger_arrival_rate=2.0,
    large_queue_threshold=8,
    wait_time_threshold=2,
    fuel_cost_move=1.5,
    fuel_cost_wait=0.4,
    new_stop_bonus=1.5,
    idle_camping_penalty=1.0,
    camping_grace_steps=0,
    nearby_queue_ignore_penalty=2.5,
    repeat_stop_penalty=0.8,
    high_queue_reward_threshold=5,
    high_queue_visit_bonus=3.0,
    reward_clip=15.0,
    demand_profile="peak_hour",
)

TASKS: Dict[str, TaskConfig] = {}

# Generate 10 subtasks for each difficulty level
for i in range(1, 11):
    # Easy subtasks
    t1 = copy.deepcopy(_TASK_EASY_TEMPLATE)
    t1.name = f"task1_{i}"
    t1.description = f"Easy variant {i}"
    t1.seed = 42 + i
    t1.passenger_arrival_rate += (i - 1) * 0.05
    TASKS[t1.name] = t1
    globals()[t1.name] = t1
    __all__.append(t1.name)

    # Medium subtasks
    t2 = copy.deepcopy(_TASK_MEDIUM_TEMPLATE)
    t2.name = f"task2_{i}"
    t2.description = f"Medium variant {i}"
    t2.seed = 42 + i
    t2.passenger_arrival_rate += (i - 1) * 0.1
    TASKS[t2.name] = t2
    globals()[t2.name] = t2
    __all__.append(t2.name)

    # Hard subtasks
    t3 = copy.deepcopy(_TASK_HARD_TEMPLATE)
    t3.name = f"task3_{i}"
    t3.description = f"Hard variant {i}"
    t3.seed = 42 + i
    t3.passenger_arrival_rate += (i - 1) * 0.15
    TASKS[t3.name] = t3
    globals()[t3.name] = t3
    __all__.append(t3.name)

# Legacy aliases for compatibility
TASK_EASY = TASKS["task1_1"]
TASK_MEDIUM = TASKS["task2_1"]
TASK_HARD = TASKS["task3_1"]

def get_task(name: str) -> TaskConfig:
    key = name.lower().strip().replace("_", "")
    
    # Map legacy names to specific subtasks
    legacy_map = {
        "easy": "task1_1",
        "medium": "task2_1",
        "hard": "task3_1",
        "task1": "task1_1",
        "task2": "task2_1",
        "task3": "task3_1",
    }
    
    # If it's something like "task11" -> "task1_1"
    if key in ["task11", "task1_1"]: return TASKS["task1_1"]
    if key in ["task21", "task2_1"]: return TASKS["task2_1"]
    if key in ["task31", "task3_1"]: return TASKS["task3_1"]
    
    # Generic mapping for task1_2, etc if they were sent as task12
    for prefix in ["task1", "task2", "task3"]:
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix):]
            if suffix.isdigit():
                possible_key = f"{prefix}_{suffix}"
                if possible_key in TASKS:
                    return TASKS[possible_key]

    key = legacy_map.get(key, name.lower().strip())
    if key not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Choose from: {list(TASKS.keys())}")
    return TASKS[key]
