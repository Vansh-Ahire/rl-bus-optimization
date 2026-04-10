# OpenEnv Environment Evaluation Rubric

This document provides a systematic scoring framework for evaluating OpenEnv environment submissions. It serves as both a reviewer guide and a self-validation tool for developers.

---

## **1. Real-World Utility (30%)**
**Weight**: 30 Points | **Focus**: Practicality, depth, and domain relevance.

| Tier | Score Range | Description |
| :--- | :---: | :--- |
| **Excellent** | 26–30 | Fills a real gap in the RL/agent community; immediate value for training production-grade agents. |
| **Good** | 16–25 | Valid domain with deep modeling; useful for rigorous agent evaluation and research. |
| **Shallow** | 6–15 | Correct domain but simplified modeling; lacks the complexity of real-world constraints. |
| **Toy/Artificial** | 0–5 | Game-like or toy problem with no practical application outside of basic RL testing. |

**Checkpoints**:
- [ ] Does the environment model a task humans actually do?
- [ ] Are real-world constraints (e.g., fuel, time, capacity) accurately represented?
- [ ] Is the data source (e.g., GTFS, logs) realistic?

---

## **2. Task & Grader Quality (25%)**
**Weight**: 25 Points | **Focus**: Objective success measurement and difficulty progression.

| Requirement | Mandatory | Criteria |
| :--- | :---: | :--- |
| **Task Count** | Yes | Minimum 3 tasks spanning Easy → Medium → Hard. |
| **Score Range** | Yes | Graders MUST output scores strictly in `(0.0, 1.0)`. |
| **Determinism** | Yes | Graders must produce reproducible scores for the same trajectory. |
| **Frontier Challenge**| Yes | The "Hard" task must genuinely challenge state-of-the-art models (e.g., Qwen-2.5-72B). |

**Scoring Indicators**:
- **0–10**: Graders are binary or inconsistent; tasks have no clear difficulty jump.
- **11–20**: Good progression; graders use multi-metric weighted scoring.
- **21–25**: Excellent difficulty scaling; hard task requires complex multi-step reasoning.

---

## **3. Environment Design (20%)**
**Weight**: 20 Points | **Focus**: State management, observation/action space, and rewards.

**Checklist**:
- [ ] **Clean State**: `reset()` produces a fresh, independent environment state.
- [ ] **Documentation**: Action and Observation spaces are fully documented with types and ranges.
- [ ] **Reward Shaping**: Reward function provides continuous signals (partial progress), not just end-of-episode success.
- [ ] **Episode Boundaries**: Logical `done` conditions prevent infinite loops or premature termination.

**Scoring Indicators**:
- **0–8**: Sparse rewards; confusing observation space; inconsistent `done` flags.
- **9–15**: Meaningful partial rewards; well-structured Pydantic models; stable state.
- **16–20**: High-fidelity simulation; sophisticated reward shaping that prevents "camping" or exploitation.

---

## **4. Code Quality & Spec Compliance (15%)**
**Weight**: 15 Points | **Focus**: Standards, deployment, and reproducibility.

**Mandatory Gates (Binary Pass/Fail)**:
- [ ] `openenv validate` passes with zero errors.
- [ ] `docker build` succeeds and `docker run` starts a responsive server on port 7860.
- [ ] Hugging Face Space is tagged `openenv` and responds to API pings.
- [ ] `inference.py` runs and reproduces baseline scores using the OpenAI client.

**Scoring Indicators**:
- **0–5**: Fails one or more mandatory gates.
- **6–10**: Passes all gates; code is functional but lacks comments/structure.
- **11–15**: Production-quality code; typed throughout; clean separation of concerns (app, env, grader).

---

## **5. Creativity & Novelty (10%)**
**Weight**: 10 Points | **Focus**: Innovation and engagement.

**Indicators**:
- **Unseen Domain**: Models a problem space not yet represented in the OpenEnv ecosystem.
- **Innovative Mechanics**: Uses unique state transitions or interaction patterns.
- **XAI Integration**: Provides reasoning tokens or "Thought-Aware" UI/UX features.
- **Engaging Visuals**: Dashboard provides high-quality real-time telemetry and insights.

**Scoring Indicators**:
- **0–3**: Standard implementation of a common problem.
- **4–7**: Creative approach to a known domain or interesting reward mechanics.
- **8–10**: Truly novel domain or highly innovative mechanics that set a new standard for OpenEnv.

---

## **Final Score Calculation**
`Final Score = (Utility * 0.30) + (TaskQuality * 0.25) + (Design * 0.20) + (CodeQuality * 0.15) + (Creativity * 0.10)`
