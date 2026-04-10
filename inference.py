"""
OpenEnv baseline inference script.

Runs an agent on all three task difficulty tiers and prints reproducible
scores with structured logging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from openai import OpenAI

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_KEY = HF_TOKEN

GLOBAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "1200"))  # 20 minutes

# Diagnostic helper: print to stderr to avoid breaking validator parsing
def dprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

from environment import BusRoutingEnv
from tasks import TASKS, TaskConfig, get_task
from grader import _grade_task


# ---------------------------------------------------------------------------
# Structured Logging (MANDATORY)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Watchdog timer
# ---------------------------------------------------------------------------

def _start_watchdog(timeout_seconds: int) -> None:
    def _watchdog():
        time.sleep(timeout_seconds)
        dprint(f"\n[TIMEOUT] Global timeout of {timeout_seconds}s reached.")
        os._exit(1)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Mock LLM agent
# ---------------------------------------------------------------------------

class MockLLMAgent:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        fuel = float(obs[1])
        q0, q1, q2 = float(obs[3]), float(obs[4]), float(obs[5])
        if fuel < 10.0: return 2
        if q0 >= max(q1, q2) and q0 > 2: return 2
        return 0


# ---------------------------------------------------------------------------
# OpenAI LLM agent
# ---------------------------------------------------------------------------

class OpenAIAgent:
    SYSTEM_PROMPT = (
        "RL bus agent. Obs: [pos (0-11), fuel (0-100), pax_onboard, q_curr, q_next, q_after, step].\n"
        "Actions: 0=move+pickup, 1=move+skip, 2=wait+pickup.\n"
        "Respond ONLY with a JSON object: {\"action\": 0|1|2}"
    )

    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.model = MODEL_NAME
        self._fallback = MockLLMAgent()

    def __call__(self, obs: np.ndarray) -> int:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Obs: {obs.tolist()}"},
                ],
                temperature=0.0,
                max_tokens=20,
                timeout=10.0,
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text.replace("```json", "").replace("```", ""))
            return int(data.get("action", 0))
        except Exception:
            return self._fallback(obs)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_inference(mode: str, model_path: Optional[str], episodes: int) -> None:
    _start_watchdog(GLOBAL_TIMEOUT)

    if mode == "dqn":
        from agent import DQNAgent
        model_path = model_path or "models/dqn_bus_v6_best.pt"
        agent_obj = DQNAgent.load(model_path)
        agent = lambda obs: agent_obj.act(obs, greedy=True)
    elif mode == "llm":
        agent = OpenAIAgent()
    else:
        agent = MockLLMAgent()

    task_ids = ["task1", "task2", "task3"]
    all_rewards = []
    total_steps = 0
    task_scores = []

    try:
        for tid in task_ids:
            task_cfg = TASKS[tid]
            env = task_cfg.build_env()
            
            # [START] per task
            log_start(task=tid, env="rl-bus-optimization", model=MODEL_NAME if mode == "llm" else "dqn-local")
            
            task_rewards = []
            task_steps = 0
            
            for _ in range(episodes):
                obs = env.reset().to_array()
                done = False
                step_count = 0
                
                while not done and step_count < task_cfg.max_steps:
                    action = int(agent(obs))
                    obs_model, reward_model, done, _ = env.step(action)
                    obs = obs_model.to_array()
                    
                    step_count += 1
                    task_steps += 1
                    total_steps += 1
                    reward = float(reward_model.value)
                    task_rewards.append(reward)
                    all_rewards.append(reward)
                    
                    log_step(step=task_steps, action=str(action), reward=reward, done=done, error=None)

            # Use grader for official score
            score = _grade_task(task_cfg, agent, episodes=episodes)["score"]
            task_scores.append(score)
            
            # [END] per task
            log_end(success=score >= 0.7, steps=task_steps, score=score, rewards=task_rewards)

        final_score = float(np.mean(task_scores))
        success = final_score >= 0.7

    except Exception as e:
        dprint(f"[ERROR] {e}")
        final_score = 0.05
        success = False
    # No global [END] here, it's per task now as per standard patterns



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["llm", "mock", "dqn"], default="llm")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    run_inference(args.mode, args.model_path, args.episodes)
