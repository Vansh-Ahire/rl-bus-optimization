import time
import os
from environment import BusRoutingEnv
from tasks import get_task
from agent import DQNAgent

def run_demo():
    print("\n" + "="*50)
    print("  OPENENV BUS OPTIMIZATION — LIVE DEMO")
    print("="*50 + "\n")

    task = get_task("medium")
    env = task.build_env()
    model_path = "models/dqn_bus_v5.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return
    
    agent = DQNAgent.load(model_path)
    obs_model = env.reset()
    obs = obs_model.to_array()

    for step in range(1, 11):
        action = agent.act(obs, greedy=True)
        obs_model, reward, done, info = env.step(action)
        obs = obs_model.to_array()
        
        render = env.render()
        bus_pos = render["bus_pos"]
        stops = render["stops"]
        
        # Simple ASCII Route
        route_str = ""
        for i, stop in enumerate(stops):
            char = f"[{stop['queue_len']:02d}]"
            if i == bus_pos:
                char = f"|🚌{stop['queue_len']:02d}|"
            route_str += char + " -- "
        
        print(f"Step {step:02d} | Action: {action} | Route: {route_str}")
        print(f"        | Fuel: {render['fuel']:.1f}% | Onboard: {render['onboard']} | Reward: {reward.value:+.2f}")
        print("-" * 100)
        
        if done: break
        time.sleep(0.5)

    print("\nDemo concluded successfully.")

if __name__ == "__main__":
    run_demo()
