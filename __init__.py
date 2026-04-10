"""
rl-bus-optimization: OpenEnv-compliant RL environment for bus route optimization.
"""

__version__ = "1.1.0"

# Expose key components for OpenEnv discovery
from environment import BusRoutingEnv
from tasks import TASKS, TaskConfig, get_task

# Explicitly expose grader functions for OpenEnv validator
from grader import (
    grade_task_1,
    grade_task_2,
    grade_task_3,
    grade_task_4,
    grade_task_5,
    grade_all_tasks,
)

__all__ = [
    "BusRoutingEnv",
    "TASKS",
    "TaskConfig",
    "get_task",
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
    "grade_task_4",
    "grade_task_5",
    "grade_all_tasks",
]
