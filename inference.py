"""
OpenEnv baseline inference script.

Runs an agent on all three task difficulty tiers and prints reproducible
scores with structured logging.

Usage:
    # Default: use pre-trained DQN model (completes in ~30 seconds):
    python inference.py

    # Explicitly use DQN with a specific checkpoint:
    python inference.py --mode dqn --model-path models/dqn_bus_v6_best.pt

    # Use LLM via API (requires API key, slower):
    python inference.py --mode llm

    # Use deterministic mock heuristic:
    python inference.py --mode mock

Environment variables:
    OPENAI_API_KEY  — API key for LLM mode (optional)
    MODEL_NAME      — LLM model name (default: openai/gpt-oss-120b:free)
    API_BASE_URL    — API endpoint (default: https://openrouter.ai/api/v1)
    MAX_EVAL_EPISODES — Episodes per task (default: 2)
    EVAL_TIMEOUT    — Global timeout in seconds (default: 1500 = 25 min)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from typing import Callable, Dict, Optional

import numpy as np

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API_KEY priority: Explicit OPENAI_API_KEY > HF_TOKEN
API_KEY = OPENAI_API_KEY or HF_TOKEN

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
GLOBAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "1500"))  # 25 minutes

# Diagnostic helper: print to stderr to avoid breaking validator parsing
def dprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

from environment import BusRoutingEnv, Observation, Action
from tasks import TASKS, TaskConfig, get_task
from grader import grade_all_tasks, grade_task_1, grade_task_2, grade_task_3


# ---------------------------------------------------------------------------
# Structured Logging (Mandatory Hackathon Requirement)
# ---------------------------------------------------------------------------

def log_start(**kwargs):
    """Emit [START] log with key-value pairs."""
    vals = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[START] {vals}", flush=True)


def log_step(**kwargs):
    """Emit [STEP] log with key-value pairs."""
    vals = " ".join(f"{k}={v if v is not None else 'null'}" for k, v in kwargs.items())
    print(f"[STEP] {vals}", flush=True)


def log_end(**kwargs):
    """Emit [END] log with key-value pairs."""
    payload = []
    for k, v in kwargs.items():
        if isinstance(v, (list, np.ndarray, tuple)):
            # Format as comma-separated list WITHOUT brackets/quotes for the validator
            v_str = ",".join(f"{x:.2f}" if isinstance(x, (float, np.float32)) else str(x) for x in v)
        else:
            v_str = str(v)
        payload.append(f"{k}={v_str}")
    vals = " ".join(payload)
    print(f"[END] {vals}", flush=True)


# ---------------------------------------------------------------------------
# Watchdog timer — kills process if evaluation exceeds global timeout
# ---------------------------------------------------------------------------

def _start_watchdog(timeout_seconds: int) -> None:
    """Start a background thread that kills the process after timeout."""
    def _watchdog():
        time.sleep(timeout_seconds)
        print(f"\n[TIMEOUT] Global timeout of {timeout_seconds}s reached. Exiting.", flush=True)
        log_end(success="false", steps=0, rewards=[0.0], reason="global_timeout")
        os._exit(1)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()
    dprint(f"[INFO] Watchdog armed: {timeout_seconds}s global deadline.")


# ---------------------------------------------------------------------------
# Mock LLM agent (deterministic fallback)
# ---------------------------------------------------------------------------

class MockLLMAgent:
    """Deterministic heuristic agent — fallback when API is unavailable."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        fuel = float(obs[1])
        q0, q1, q2 = float(obs[3]), float(obs[4]), float(obs[5])
        if fuel < 10.0:
            return 2
        if q0 >= max(q1, q2) and q0 > 2:
            return 2
        if q1 >= q2:
            return 0
        return 0


# ---------------------------------------------------------------------------
# OpenAI LLM agent (with strict per-call timeout)
# ---------------------------------------------------------------------------

class OpenAIAgent:
    """Agent that queries an LLM API — used only when --mode llm is explicit."""

    SYSTEM_PROMPT = (
        "RL bus agent. Obs: [pos (0-11), fuel (0-100), pax_onboard, q_curr, q_next, q_after, step].\n"
        "Actions: 0=move+pickup, 1=move+skip, 2=wait+pickup.\n"
        "Goals: Max pickups, min wait, save fuel.\n"
        "Respond ONLY: {\"action\": 0|1|2}"
    )

    def __init__(self, temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        self.model = MODEL_NAME
        self.temperature = temperature
        self._fallback = MockLLMAgent()

    def __call__(self, obs: np.ndarray) -> int:
        user_msg = (
            f"Current observation: {obs.tolist()}\n"
            f"Choose your action (0, 1, or 2). Respond ONLY with JSON."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=20,
                timeout=8.0,  # Strict 8s timeout per call
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)
            action = int(data.get("action", 0))
            if action not in (0, 1, 2):
                action = 0
            return action
        except Exception as e:
            dprint(f"[WARN] LLM call failed ({type(e).__name__}), using heuristic fallback")
            return self._fallback(obs)


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agent(mode: str, model_path: Optional[str] = None) -> Callable[[np.ndarray], int]:
    """
    Build the agent callable.

    Modes:
        dqn   — Pre-trained DQN checkpoint (DEFAULT — fast, local, reliable)
        llm   — OpenAI-compatible API
        mock  — Deterministic heuristic
    """
    if mode == "dqn":
        from agent import DQNAgent

        if model_path is None:
            # Try multiple known model paths
            candidates = [
                "models/dqn_bus_v6_best.pt",
                "models/dqn_bus_v6.pt",
                "models/dqn_bus.pt",
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    model_path = candidate
                    break

        if model_path is None or not os.path.isfile(model_path):
            dprint(f"[WARN] No DQN model found. Falling back to mock agent.")
            return MockLLMAgent()

        dprint(f"[INFO] Loading DQN model from '{model_path}'")
        agent = DQNAgent.load(model_path)
        return lambda obs: agent.act(obs, greedy=True)

    if mode == "llm":
        # Strict token check for LLM mode
        if not API_KEY:
            raise ValueError("HF_TOKEN or OPENAI_API_KEY environment variable is required for LLM mode")

        dprint("[INFO] Using LLM API agent.")
        return OpenAIAgent()

    # Default: mock
    dprint("[INFO] Using mock (heuristic) agent.")
    return MockLLMAgent()


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(mode: str, model_path: Optional[str], episodes: int) -> Dict:
    """Run inference across all three tasks with trajectory-based logging."""

    # Start the watchdog timer
    _start_watchdog(GLOBAL_TIMEOUT)

    agent = build_agent(mode, model_path)

    dprint(f"\n{'=' * 60}")
    dprint("  OpenEnv Bus Routing - Inference")
    dprint(f"{'=' * 60}")
    dprint(f"  Mode     : {mode}")
    dprint(f"  Episodes : {episodes}")
    dprint(f"  Timeout  : {GLOBAL_TIMEOUT}s")
    dprint(f"{'=' * 60}\n")

    t0 = time.time()

    all_rewards = []
    total_steps = 0
    results = {}
    task_keys = [
        ("task_1", "easy"), 
        ("task_2", "medium"), 
        ("task_3", "hard"),
        ("task_4", "medium"),
        ("task_5", "hard")
    ]
    
    # Use try...finally to guarantee [END] log
    try:
        # Mandatory: [START] log
        log_start(task=mode, env="rl-bus-optimization", model=MODEL_NAME if mode == "llm" else f"dqn-local")

        for i, (report_key, _difficulty) in enumerate(task_keys):
            dprint(f"[INFO] Evaluating {report_key} task...")
            task_cfg = TASKS[report_key]
            env = task_cfg.build_env()
            
            # Run evaluation episodes for this task
            for ep in range(episodes):
                obs_model = env.reset()
                obs = obs_model.to_array()
                done = False
                step_idx = 1
                
                while not done:
                    action = int(agent(obs))
                    obs_model, reward_model, done, info = env.step(action)
                    obs = obs_model.to_array()
                    
                    # Mandatory: [STEP] log per environment step 
                    # Precision: 2 decimal places for rewards
                    log_step(
                        step=total_steps + step_idx,
                        action=action,
                        reward=f"{reward_model.value:.2f}",
                        done="true" if done else "false",
                        error="null"
                    )
                    
                    all_rewards.append(reward_model.value)
                    step_idx += 1
                    if step_idx > task_cfg.max_steps:
                        done = True
                
                total_steps += (step_idx - 1)

            # Standard grader metrics
            from grader import _grade_task
            report = _grade_task(task_cfg, agent, episodes=episodes)
            results[report_key] = report

        # Calculate aggregate score (uniformly over tasks)
        scores = [results[k]["score"] for k, _ in task_keys]
        final_score = float(np.mean(scores))
        
        SUCCESS_THRESHOLD = 0.7
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        dprint(f"[ERROR] Inference crashed: {e}")
        final_score = 0.0
        success = False
        raise
    finally:
        log_end(
            success="true" if success else "false",
            steps=total_steps,
            rewards=all_rewards
        )

    elapsed = time.time() - t0

    # Pretty print summary (to stderr)
    dprint(f"\n{'=' * 55}")
    dprint(f"  AGGREGATE SCORE : {final_score:.4f}")
    dprint(f"  Success         : {success}")
    dprint(f"  Total Steps     : {total_steps}")
    dprint(f"  Time elapsed    : {elapsed:.2f}s")
    dprint(f"{'=' * 55}\n")

    results["aggregate_score"] = final_score
    results["success"] = success
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenEnv baseline inference — runs agent on all tasks"
    )
    p.add_argument(
        "--mode",
        choices=["llm", "mock", "dqn"],
        default="llm",  # DEFAULT: LLM — mandatory for proxy monitoring
        help="Agent mode: 'dqn' (pre-trained model), 'llm' (API, DEFAULT), or 'mock' (heuristic).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to DQN model checkpoint (only used in dqn mode).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=int(os.getenv("MAX_EVAL_EPISODES", 1)),
        help="Number of evaluation episodes per task.",
    )
    args = p.parse_args()

    run_inference(args.mode, args.model_path, args.episodes)


if __name__ == "__main__":
    main()
