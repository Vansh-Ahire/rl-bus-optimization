"""
Enhanced training script for the Double DQN (DDQN) bus routing agent.

Upgrades:
- Best-model saving (tracks max cumulative reward)
- Expanded metric tracking (Loss, Avg Q-Values)
- Improved terminal telemetry
- Multi-task support with OpenEnv compliance
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from environment import BusRoutingEnv
from agent import DQNAgent, DQNConfig
from tasks import get_task


def train(
    task_name: str = "medium",
    episodes: int = 200,             # Increased default for better convergence
    seed: int = 0,
    model_out: str = "models/dqn_bus.pt",
    metrics_out: str = "models/training_metrics.csv",
) -> Dict[str, List[float]]:
    """Train a DDQN agent on the specified task and save the best model."""
    task_cfg = get_task(task_name)
    task_cfg.seed = seed
    env = task_cfg.build_env()

    # Initialize Agent with optimized Hackathon-level config
    agent = DQNAgent(env.obs_size, env.num_actions, config=DQNConfig(), seed=seed)

    history: Dict[str, List[float]] = {
        "reward": [], 
        "avg_wait": [], 
        "fuel_used": [],
        "loss": [],
        "epsilon": []
    }

    best_reward = -float("inf")
    best_model_path = model_out.replace(".pt", "_best.pt")

    print(f"🚀 Training Hackathon-Level DDQN on task: {task_cfg.name}")
    print(f"   Stops: {task_cfg.num_stops} | Max Steps: {task_cfg.max_steps} | Capacity: {task_cfg.bus_capacity}")
    print(f"   Episodes: {episodes} | Seed: {seed}")
    print("-" * 60)

    for ep in range(1, int(episodes) + 1):
        obs_model = env.reset()
        obs = obs_model.to_array()
        done = False
        
        episode_losses = []
        
        while not done:
            # select_action uses the new internal pipeline (preprocess -> select)
            action = agent.act(obs, greedy=False)
            obs_model, reward_model, done, _info = env.step(action)
            obs2 = obs_model.to_array()
            
            agent.observe(obs, action, reward_model.value, obs2, done)
            obs = obs2
            
            if agent.can_train():
                metrics = agent.train_step()
                if not np.isnan(metrics["loss"]):
                    episode_losses.append(metrics["loss"])

        # Episode stats calculation
        avg_wait = (
            env.total_wait_time_picked / env.total_picked
            if env.total_picked > 0
            else 20.0 # Penalty/default for no pickups
        )
        total_reward = float(env.total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        history["reward"].append(total_reward)
        history["avg_wait"].append(float(avg_wait))
        history["fuel_used"].append(float(env.total_fuel_used))
        history["loss"].append(float(avg_loss))
        history["epsilon"].append(agent.epsilon())
        
        agent.on_episode_end()

        # [BEST MODEL SAVING]
        if total_reward > best_reward and ep > 20: 
            best_reward = total_reward
            os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)
            agent.save(best_model_path)
            # print(f"   [New Best!] Ep {ep:03d} | Reward: {total_reward:.2f}")

        # Logging periodic status
        if ep % 20 == 0 or ep == 1 or ep == episodes:
            print(
                f"ep={ep:03d} | rew={total_reward:7.1f} | wait={avg_wait:5.2f} | "
                f"fuel={env.total_fuel_used:5.1f} | loss={avg_loss:6.4f} | eps={agent.epsilon():.3f}"
            )

    # Save final model
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    agent.save(model_out)
    print(f"\n✅ Training Complete.")
    print(f"   Final Model: {model_out}")
    print(f"   Best Model:  {best_model_path} (Reward: {best_reward:.2f})")

    if metrics_out:
        os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            f.write("episode,total_reward,avg_wait_time,fuel_used,loss,epsilon\n")
            for i in range(len(history["reward"])):
                f.write(f"{i+1},{history['reward'][i]},{history['avg_wait'][i]},"
                        f"{history['fuel_used'][i]},{history['loss'][i]},{history['epsilon'][i]}\n")
        print(f"   Metrics:     {metrics_out}")

    return history


def main() -> None:
    p = argparse.ArgumentParser(description="Train Double DQN agent on an OpenEnv task")
    p.add_argument("--task", type=str, default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-out", type=str, default="models/dqn_bus_v6.pt")
    p.add_argument("--metrics-out", type=str, default="models/training_metrics_v6.csv")
    args = p.parse_args()

    train(
        task_name=args.task,
        episodes=args.episodes,
        seed=args.seed,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
    )


if __name__ == "__main__":
    main()
