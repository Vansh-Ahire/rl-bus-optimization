"""
Multi-task configuration for the OpenEnv bus routing environment.

Three difficulty tiers — Easy, Medium, Hard — share the same
``BusRoutingEnv`` class but differ in the number of stops, passenger
demand, fuel constraints, and penalty intensity.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

from environment import BusRoutingEnv

# Explicitly export task configurations for OpenEnv detection
__all__ = [
    "TaskConfig",
    "task_1",
    "task_2", 
    "task_3",
    "task_4",
    "task_5",
    "task_6",
    "task_7",
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
    name="task_easy",
    description="Easy template",
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
    name="task_medium",
    description="Medium template",
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
    name="task_hard",
    description="Hard template",
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

task_1 = copy.deepcopy(_TASK_EASY_TEMPLATE)
task_1.name = "task_1"
task_1.description = "Easy task 1"

task_2 = copy.deepcopy(_TASK_MEDIUM_TEMPLATE)
task_2.name = "task_2"
task_2.description = "Medium task 2"

task_3 = copy.deepcopy(_TASK_HARD_TEMPLATE)
task_3.name = "task_3"
task_3.description = "Hard task 3"

task_4 = copy.deepcopy(_TASK_MEDIUM_TEMPLATE)
task_4.name = "task_4"
task_4.description = "Medium task 4 (Alternative Seed)"
task_4.seed = 99

task_5 = copy.deepcopy(_TASK_HARD_TEMPLATE)
task_5.name = "task_5"
task_5.description = "Hard task 5 (Extreme Peak)"
task_5.passenger_arrival_rate = 2.5
task_5.seed = 123

task_6 = copy.deepcopy(_TASK_HARD_TEMPLATE)
task_6.name = "task_6"
task_6.description = "Very Hard - Large Network (20 stops)"
task_6.num_stops = 20
task_6.num_buses = 2
task_6.max_steps = 250
task_6.fuel_start = 75.0
task_6.passenger_arrival_rate = 2.2
task_6.seed = 456
task_6.large_queue_threshold = 7
task_6.wait_time_threshold = 2
task_6.fuel_cost_move = 1.6
task_6.fuel_cost_wait = 0.45
task_6.new_stop_bonus = 1.6
task_6.idle_camping_penalty = 1.2
task_6.nearby_queue_ignore_penalty = 2.8
task_6.repeat_stop_penalty = 0.9
task_6.high_queue_reward_threshold = 4
task_6.high_queue_visit_bonus = 3.5
task_6.reward_clip = 18.0

task_7 = copy.deepcopy(_TASK_HARD_TEMPLATE)
task_7.name = "task_7"
task_7.description = "Extreme - Mega Network (25 stops)"
task_7.num_stops = 25
task_7.num_buses = 2
task_7.max_steps = 300
task_7.fuel_start = 70.0
task_7.passenger_arrival_rate = 2.8
task_7.seed = 789
task_7.large_queue_threshold = 6
task_7.wait_time_threshold = 1
task_7.fuel_cost_move = 1.8
task_7.fuel_cost_wait = 0.5
task_7.new_stop_bonus = 1.8
task_7.idle_camping_penalty = 1.5
task_7.nearby_queue_ignore_penalty = 3.0
task_7.repeat_stop_penalty = 1.0
task_7.high_queue_reward_threshold = 3
task_7.high_queue_visit_bonus = 4.0
task_7.reward_clip = 20.0

TASKS: Dict[str, TaskConfig] = {
    "task_1": task_1,
    "task_2": task_2,
    "task_3": task_3,
    "task_4": task_4,
    "task_5": task_5,
    "task_6": task_6,
    "task_7": task_7,
}

TASK_EASY = task_1
TASK_MEDIUM = task_2
TASK_HARD = task_3


def get_task(name: str) -> TaskConfig:
    key = name.lower().strip()
    legacy_map = {
        "easy": "task_1",
        "medium": "task_2",
        "hard": "task_3",
        "task1": "task_1",
        "task2": "task_2",
        "task3": "task_3",
        "task_11": "task_2",
        "task_21": "task_3",
    }
    key = legacy_map.get(key, key)
    if key not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Choose from: {list(TASKS.keys())}")
    return TASKS[key]
