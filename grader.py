"""
Deterministic per-task graders for the OpenEnv bus routing environment.

Each ``grade_task_X`` function:
    1. Creates the task environment from ``tasks.py``.
    2. Runs the agent over multiple episodes.
    3. Compares against heuristic baselines.
    4. Returns a normalised **score in [0.0, 1.0]**.

Scoring considers:
    • Average passenger wait time
    • Cumulative reward
    • Fuel efficiency (pickups per fuel unit)
    • Stop coverage (fraction of stops visited)
    • Route balance (normalised entropy of visit distribution)
    • Anti-camping (penalises over-concentration at a single stop)
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from environment import BusRoutingEnv
from tasks import TASKS, TaskConfig

# Explicitly export grader functions for OpenEnv detection
__all__ = [
    "grade_task1",
    "grade_task2", 
    "grade_task3",
    "grade_all_tasks",
]


# ---------------------------------------------------------------------------
# Heuristic baselines
# ---------------------------------------------------------------------------

def random_policy(_obs: np.ndarray, num_actions: int = 3) -> int:
    return int(np.random.randint(0, num_actions))


def greedy_baseline_policy(obs: np.ndarray) -> int:
    """
    Simple heuristic:
        - If current stop queue is large → wait & pick up
        - Else if next stop queue >= current → move + pickup
        - Else skip
    obs = [pos, fuel, onboard, q0, q1, q2, time]
    """
    q0, q1 = obs[3], obs[4]
    if q0 >= 8:
        return 2  # wait
    if q1 >= q0:
        return 0  # move+pickup
    return 1  # move+skip


def highest_queue_first_policy(obs: np.ndarray) -> int:
    """
    Stronger heuristic — serve the largest nearby queue:
        - If current queue >= both neighbours → wait
        - Else → move + pickup
    """
    q0, q1, q2 = float(obs[3]), float(obs[4]), float(obs[5])
    if q0 >= max(q1, q2):
        return 2
    return 0


def or_tools_greedy_policy(obs: np.ndarray) -> int:
    """
    OR-Tools-like greedy routing heuristic:
        - If current queue > 5: wait (action=2)
        - Else: move to stop with highest queue (action=0 or 1)
        - Simulates distance + demand based routing
    """
    q0, q1, q2 = float(obs[3]), float(obs[4]), float(obs[5])
    fuel = float(obs[1])
    
    if q0 > 5:
        return 2
    if fuel < 20:
        return 1
    if q1 >= q2:
        return 0
    return 1


def mpc_baseline_policy(obs: np.ndarray) -> int:
    """
    Model Predictive Control baseline:
        - Look ahead with fuel consideration
        - If fuel low (<20): move+skip (conserve fuel)
        - If fuel high (>50): aggressive wait+pickup
    """
    q0, q1, q2 = float(obs[3]), float(obs[4]), float(obs[5])
    fuel = float(obs[1])
    
    if fuel < 20:
        if q0 > 8:
            return 2
        return 1
    if fuel > 50:
        if q0 >= max(q1, q2):
            return 2
        return 0
    if q0 > 6:
        return 2
    if q1 > q0:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_eval(
    env: BusRoutingEnv,
    policy: Callable[[np.ndarray], int],
    episodes: int = 20,
) -> Dict[str, float]:
    rewards: List[float] = []
    waits: List[float] = []
    fuels: List[float] = []
    covers: List[float] = []
    entropies: List[float] = []
    max_stop_fracs: List[float] = []
    picks: List[float] = []

    for _ in range(int(episodes)):
        m = env.run_episode(policy_fn=policy)
        rewards.append(m["total_reward"])
        waits.append(m["avg_wait_time"])
        fuels.append(m["fuel_used"])
        covers.append(m["stop_coverage"])
        entropies.append(m.get("route_entropy", 0.0))
        max_stop_fracs.append(m.get("max_stop_fraction", 1.0))
        picks.append(m["passengers_picked"])

    waits_safe = [w if np.isfinite(w) else 50.0 for w in waits]
    return {
        "avg_wait_time": float(np.mean(waits_safe)),
        "total_reward": float(np.mean(rewards)),
        "fuel_efficiency": float(np.mean(picks) / (np.mean(fuels) + 1e-6)),
        "stop_coverage": float(np.mean(covers)),
        "route_entropy": float(np.mean(entropies)),
        "max_stop_fraction": float(np.mean(max_stop_fracs)),
        "avg_passengers_picked": float(np.mean(picks)),
    }


def _add_statistical_tests(
    env: BusRoutingEnv,
    agent_policy: Callable[[np.ndarray], int],
    baseline_policy: Callable[[np.ndarray], int],
    episodes: int = 20,
) -> Dict[str, float]:
    """Perform statistical significance testing between agent and baseline."""
    if not SCIPY_AVAILABLE:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_improvement": 0.0,
            "confidence_interval": (0.0, 0.0),
            "statistical_significance": "scipy not available"
        }
    
    agent_rewards = []
    baseline_rewards = []
    
    for _ in range(episodes):
        m_agent = env.run_episode(policy_fn=agent_policy)
        m_baseline = env.run_episode(policy_fn=baseline_policy)
        agent_rewards.append(m_agent["total_reward"])
        baseline_rewards.append(m_baseline["total_reward"])
    
    t_statistic, p_value = stats.ttest_ind(agent_rewards, baseline_rewards)
    mean_agent = np.mean(agent_rewards)
    mean_baseline = np.mean(baseline_rewards)
    mean_improvement = ((mean_agent - mean_baseline) / abs(mean_baseline + 1e-6)) * 100
    diff = np.array(agent_rewards) - np.array(baseline_rewards)
    ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    significance = "p < 0.05 [PASS]" if p_value < 0.05 else "p >= 0.05"
    
    return {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "mean_improvement": float(mean_improvement),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "statistical_significance": significance
    }


def _score_0_1(metrics: Dict[str, float], baseline: Dict[str, float]) -> float:
    """
    Weighted score normalised to **[0.0, 1.0]**.

    Weight distribution:
        wait-time improvement  30 %
        reward improvement     35 %
        fuel efficiency         5 %
        stop coverage          15 %
        route balance          10 %
        anti-camping            5 %
    """
    wait_impr = (baseline["avg_wait_time"] - metrics["avg_wait_time"]) / max(
        baseline["avg_wait_time"], 1e-6
    )
    rew_impr = (metrics["total_reward"] - baseline["total_reward"]) / (
        abs(baseline["total_reward"]) + 1e-6
    )

    wait_score = float(np.clip(wait_impr, -1.0, 1.0) * 0.5 + 0.5)
    rew_score = float(np.clip(rew_impr, -1.0, 1.0) * 0.5 + 0.5)
    fuel_score = float(np.clip(metrics["fuel_efficiency"] / 0.25, 0.0, 1.0))
    cov_score = float(np.clip(metrics["stop_coverage"], 0.0, 1.0))
    bal_score = float(np.clip(metrics.get("route_entropy", 0.0), 0.0, 1.0))
    anti_camp_score = float(
        np.clip(1.0 - metrics.get("max_stop_fraction", 1.0), 0.0, 1.0)
    )

    final = (
        0.30 * wait_score
        + 0.35 * rew_score
        + 0.05 * fuel_score
        + 0.15 * cov_score
        + 0.10 * bal_score
        + 0.05 * anti_camp_score
    )
    if not np.isfinite(final):
        return 0.15
    # Strict (0, 1) range: ensures score is never 0.0 and never 1.0
    return float(np.clip(final, 0.05, 0.95))


# ---------------------------------------------------------------------------
# Per-task grading (deterministic) — core OpenEnv requirement
# ---------------------------------------------------------------------------

def _grade_task(
    task_cfg: TaskConfig,
    agent_policy: Callable[[np.ndarray], int],
    episodes: int = 20,
) -> Dict:
    """Generic grader — used by all grade_task_X functions with statistical tests and multiple baselines."""
    env = task_cfg.build_env()

    rl_metrics = _run_eval(env, policy=agent_policy, episodes=episodes)
    baseline_metrics = _run_eval(
        env, policy=greedy_baseline_policy, episodes=episodes
    )
    random_metrics = _run_eval(
        env,
        policy=lambda obs: random_policy(obs, env.num_actions),
        episodes=episodes,
    )
    hqf_metrics = _run_eval(
        env, policy=highest_queue_first_policy, episodes=episodes
    )
    or_tools_metrics = _run_eval(
        env, policy=or_tools_greedy_policy, episodes=episodes
    )
    mpc_metrics = _run_eval(
        env, policy=mpc_baseline_policy, episodes=episodes
    )

    stats_results = _add_statistical_tests(
        env, agent_policy, greedy_baseline_policy, episodes=episodes
    )

    score = _score_0_1(rl_metrics, baseline_metrics)

    return {
        "task": task_cfg.name,
        "difficulty": task_cfg.difficulty,
        "score": score,
        "rl_agent": rl_metrics,
        "baseline_greedy": baseline_metrics,
        "baseline_random": random_metrics,
        "baseline_highest_queue_first": hqf_metrics,
        "baseline_or_tools": or_tools_metrics,
        "baseline_mpc": mpc_metrics,
        "statistical_tests": stats_results,
    }


# ---------------------------------------------------------------------------
# Per-task grading
# ---------------------------------------------------------------------------
# We explicitly define these to ensure the OpenEnv evaluator can find them via reflection.

def grade_task1(agent_policy: Callable[[np.ndarray], int], episodes: int = 20) -> float:
    """
    Grade agent performance on task1 (Easy difficulty).
    """
    return float(_grade_task(TASKS["task1"], agent_policy, episodes)["score"])


def grade_task2(agent_policy: Callable[[np.ndarray], int], episodes: int = 20) -> float:
    """
    Grade agent performance on task2 (Medium difficulty).
    """
    return float(_grade_task(TASKS["task2"], agent_policy, episodes)["score"])


def grade_task3(agent_policy: Callable[[np.ndarray], int], episodes: int = 20) -> float:
    """
    Grade agent performance on task3 (Hard difficulty).
    """
    return float(_grade_task(TASKS["task3"], agent_policy, episodes)["score"])


def grade_all_tasks(
    agent_policy: Callable[[np.ndarray], int],
    episodes: int = 20,
) -> Dict:
    """Run explicit task graders and return combined results for all 3 tasks."""
    results = {}
    total_score = 0.0

    for i in range(1, 4):
        task_id = f"task{i}"
        report = _grade_task(TASKS[task_id], agent_policy, episodes)
        results[task_id] = report
        total_score += report["score"]

    aggregate = total_score / 3.0

    return {
        **results,
        "aggregate_score": float(np.clip(aggregate, 0.05, 0.95)),
        "task_ids": list(results.keys()),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from agent import DQNAgent

    p = argparse.ArgumentParser(description="OpenEnv Bus Routing — Programmatic Grader")
    p.add_argument("--model-path", type=str, default="models/dqn_bus.pt")
    p.add_argument("--episodes", type=int, default=int(os.getenv("MAX_EVAL_EPISODES", 5)))
    args = p.parse_args()

    agent = DQNAgent.load(args.model_path)
    policy = lambda obs: agent.act(obs, greedy=True)  # noqa: E731

    report = grade_all_tasks(policy, episodes=args.episodes)

    print("=" * 60)
    print("  OpenEnv Programmatic Grade Report")
    print("=" * 60)

    for task_key in report.get("task_ids", []):
        tr = report[task_key]
        print(f"\n{'-' * 50}")
        print(f"  {tr['task']} ({tr['difficulty']})  -  score: {tr['score']:.4f}")
        print(f"{'-' * 50}")
        
        stats = tr.get("statistical_tests", {})
        if stats:
            print(f"  [Statistical Tests]")
            print(f"    p_value: {stats.get('p_value', 0.0):.4f}")
            print(f"    t_statistic: {stats.get('t_statistic', 0.0):.4f}")
            print(f"    mean_improvement: {stats.get('mean_improvement', 0.0):.2f}%")
            print(f"    significance: {stats.get('statistical_significance', 'N/A')}")

    print(f"\n{'=' * 60}")
    print(f"  Aggregate score (0.05 - 0.95): {report['aggregate_score']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
