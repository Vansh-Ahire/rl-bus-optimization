---
title: OpenEnv Bus Routing
emoji: 🚌
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - transport-optimization
  - dueling-dqn
  - gtfs
---

<div align="center">

# 🚌 OpenEnv Bus Routing Optimizer

### Dueling DDQN + Prioritized Experience Replay for Urban Transit

**Real data. Real constraints. Real RL.**

[![Built on OpenEnv](https://img.shields.io/badge/Built%20on-OpenEnv-blue)](https://github.com/openenv/openenv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Algorithm](https://img.shields.io/badge/Algorithm-Dueling%20DDQN%20%2B%20PER-purple)](https://arxiv.org/abs/1511.06581)
[![Data](https://img.shields.io/badge/Data-GTFS%20Calibrated-orange)](https://transitfeeds.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

### 🚀 [VIEW LIVE DEMO ON HUGGING FACE](https://huggingface.co/spaces/voldemort6996/rl-bus-optimizer)

</div>

---

## 🎯 Problem Statement

Urban public transit faces a fundamental optimization tension: **Service Quality vs. Operational Cost**.

In dynamic-demand scenarios (micro-transit, campus shuttles, last-mile connectivity), fixed schedules are inherently suboptimal. A bus that waits too long at a sparse stop causes downstream passenger anger; one that moves constantly without picking up wastes fuel.

**This project trains a Deep RL agent to act as an intelligent dispatcher**, dynamically deciding when to wait, move, or skip — all under strict fuel constraints and with real-world demand patterns calibrated from Indian city transit (GTFS) data.

### Key Results

| Metric | Greedy Baseline | **Our Trained DQN** | Improvement |
|--------|----------------|---------------------|-------------|
| Avg Wait Time | ~6.5 steps | **~3.2 steps** | **↓ 51%** |
| Total Reward | 115.0 | **185.0** | **↑ 61%** |
| Fuel Efficiency | 0.18 pax/fuel | **0.31 pax/fuel** | **↑ 72%** |
| Overall Score | ~0.50 | **~0.92** | **↑ 84%** |
| **Neural Load** | N/A | **Thinking-Aware** | **XAI+** |

*Evaluated over 20 episodes on Task Medium (10-stop weekday demand profile).*

---

## 📊 Performance Visualizations

### Training Progress
![Training Curves](docs/images/training_curves.png)

The RL agent (Dueling DDQN + PER) significantly outperforms both greedy and random baselines, achieving 61% improvement in cumulative reward over training episodes.

### Task Difficulty Performance
![Task Difficulty Heatmap](docs/images/task_difficulty_heatmap.png)

Agent performance scales appropriately with task difficulty, maintaining strong performance (70%+ score) even on extreme-scale tasks with 25 stops.

### Baseline Comparison
![Metrics Comparison](docs/images/metrics_comparison.png)

Comprehensive comparison across key metrics shows the agent outperforms all baselines by 15-40% on wait time, reward, fuel efficiency, and coverage.

### Route Distribution Analysis
![Stop Visitation Heatmap](docs/images/stop_visitation_heatmap.png)

The RL agent demonstrates balanced route coverage compared to greedy baselines which tend to concentrate on high-demand stops, leading to better overall service quality.

---

**To regenerate these charts**, run:
```bash
python generate_visualizations.py
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPENENV BUS OPTIMIZER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Dashboard   │◄──►│  Endpoints   │◄──►│  Panel + CoT  │      │
│  │ (server/app) │    │ (/reset,etc) │    │ (Insight XAI)│      │
│  └──────┬───────┘    └──────────────┘    └──────────────┘      │
│         │                                                       │
│  ┌──────▼───────────────────────────────────────────────┐      │
│  │  BusRoutingEnv  (OpenEnv Gymnasium Interface)        │      │
│  │                                                       │      │
│  │  POST /reset → Observation (Pydantic)                │      │
│  │  POST /step  → (Observation, Reward, done, info)    │      │
│  │  GET  /state → Full environment state                │      │
│  │                                                       │      │
│  │  Demand: GTFS-Calibrated (Pune PMPML / Mumbai BEST)  │      │
│  │  Constraints: Fuel, Capacity, Anti-Camp, Coverage     │      │
│  └──────┬───────────────────────────────────────────────┘      │
│         │                                                       │
│  ┌──────▼───────────────────────────────────────────────┐      │
│  │  Dueling Double DQN Agent + PER                      │      │
│  │                                                       │      │
│  │  Q(s,a) = V(s) + A(s,a) - mean(A)                   │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  tasks.py    │    │  grader.py   │    │  inference.py │      │
│  │  3 Tiers     │    │  Log Markers │    │  Strict Tags │      │
│  │  Easy/Med/Hd │    │ [START/END]  │    │  compliant   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  GTFS Data Layer (data/gtfs_profiles.py)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Algorithm Details

### Dueling Double DQN with Prioritized Experience Replay

Our agent combines three state-of-the-art improvements over vanilla DQN:

#### 1. Dueling Architecture (Wang et al., 2016)

The Q-network is split into two streams:

```
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
```

- **Value stream V(s)**: "How good is this state?" — learns state quality independent of actions
- **Advantage stream A(s,a)**: "How much better is action `a` vs. average?" — learns relative action benefit

#### 2. Double DQN (van Hasselt et al., 2016)

Standard DQN overestimates Q-values because it uses the same network for both selecting and evaluating actions. Double DQN decouples these.

#### 3. Prioritized Experience Replay (Schaul et al., 2016)

Instead of sampling uniformly, PER samples transitions proportional to their TD-error, accelerating learning on edge cases like fuel depletion.

---

## 🌍 Real-World Data: GTFS-Calibrated Demand

Instead of uniform synthetic arrivals, our environment uses **time-of-day demand curves** and **stop-type heterogeneity** calibrated from publicly available GTFS feeds (Pune PMPML / Mumbai BEST).

---

## 📦 OpenEnv Compliance

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| reset()/step/state API | ✅ | FastAPI endpoints for automated validation |
| Multi-task framework | ✅ | 3 tiers: task1, task2, task3 |
| Deterministic graders | ✅ | grade_task1/2/3() -> score [0.05, 0.95] |
| LLM inference support | ✅ | inference.py with OpenAI client |
| START/STEP/END logging | ✅ | Mandatory structured tags for evaluation |
| Docker containerization | ✅ | optimized Dockerfile with entry points |
| Neural Load XAI | ✅ | Real-time reasoning token tracking |

---

## 🚀 Setup & Running

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the grader
python grader.py --episodes 5

# Run the inference script (LLM mode)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py --mode llm

# Launch the dashboard + API server
python server/app.py
```

### Pre-Submission Validation

Before submitting to the hackathon, run:

```bash
python tests/FINAL_CHECK.py
```

Expected output: `SUCCESS: ALL CHECKS PASSED`

See [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) for detailed validation instructions.

## 📚 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project structure and organization
- **[VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)** - How to validate before submission
- **[docs/GRADER_FIX_SUMMARY.md](docs/GRADER_FIX_SUMMARY.md)** - Grader detection fix details
- **[docs/OPENENV_COMPLIANCE_ASSESSMENT.md](docs/OPENENV_COMPLIANCE_ASSESSMENT.md)** - OpenEnv compliance details

---

## 🔬 Research References

- **Dueling DQN**: [Wang et al., 2016](https://arxiv.org/abs/1511.06581)
- **Double DQN**: [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)
- **Prioritized Replay**: [Schaul et al., 2016](https://arxiv.org/abs/1511.05952)
- **OpenEnv**: [Meta PyTorch](https://github.com/openenv/openenv)

---

<div align="center">

**Built for the OpenEnv Hackathon 2026 — Meta PyTorch**

</div>
