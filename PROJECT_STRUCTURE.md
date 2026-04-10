# Project Structure

## Directory Layout

```
rl-bus-optimization/
├── 📁 Core Application
│   ├── __init__.py              # Package initialization with grader exports
│   ├── environment.py           # BusRoutingEnv (OpenEnv Gymnasium interface)
│   ├── agent.py                 # Dueling Double DQN implementation
│   ├── tasks.py                 # Multi-task configurations (Easy/Medium/Hard)
│   ├── grader.py                # Deterministic graders for evaluation
│   ├── inference.py             # LLM inference with structured logging
│   ├── train.py                 # Training script for DQN agent
│   ├── demonstrate.py           # Demo script for trained agent
│   └── llm_evaluator.py         # LLM-based evaluation utilities
│
├── 📁 data/
│   ├── gtfs_profiles.py         # GTFS-calibrated demand profiles
│   └── __init__.py
│
├── 📁 server/
│   ├── app.py                   # FastAPI server (OpenEnv endpoints)
│   └── __init__.py
│
├── 📁 models/
│   ├── dqn_bus_v6_best.pt       # Best trained model checkpoint
│   ├── dqn_bus_v*.pt            # Model checkpoints
│   └── training_metrics_v*.csv  # Training metrics
│
├── 📁 tests/                    # Validation & Testing Scripts
│   ├── FINAL_CHECK.py           # Quick pre-submission validation
│   ├── test_grader_detection.py # Test grader function discovery
│   ├── test_openenv_yaml.py     # Test YAML configuration
│   ├── test_validator_simulation.py # Simulate validator behavior
│   ├── test_exact_validator_flow.py # Exact validator flow simulation
│   ├── final_validation.py      # Comprehensive validation suite
│   └── PRE_SUBMIT_CHECK.py      # Pre-submission check runner
│
├── 📁 docs/                     # Documentation
│   ├── GRADER_FIX_SUMMARY.md    # Summary of grader detection fix
│   ├── OPENENV_COMPLIANCE_ASSESSMENT.md # OpenEnv compliance details
│   ├── PRE_SUBMIT_CHECKLIST.md  # Pre-submission checklist
│   ├── FINAL_VERDICT.txt        # Final validation verdict
│   ├── grader_output.txt        # Grader execution output
│   └── grader_results_final.txt # Final grader results
│
├── 📄 Configuration Files
│   ├── openenv.yaml             # OpenEnv specification
│   ├── pyproject.toml           # Python project configuration
│   ├── requirements.txt         # Python dependencies
│   ├── uv.lock                  # UV lock file
│   ├── Dockerfile               # Docker container configuration
│   └── .gitignore               # Git ignore rules
│
└── 📄 Documentation
    ├── README.md                # Main project documentation
    └── PROJECT_STRUCTURE.md    # This file
```

## Core Components

### Environment (`environment.py`)
- **BusRoutingEnv**: OpenEnv-compliant Gymnasium environment
- Implements `reset()`, `step()`, `state()` endpoints
- GTFS-calibrated demand profiles
- Fuel constraints, capacity limits, anti-camping penalties

### Agent (`agent.py`)
- **Dueling Double DQN** with Prioritized Experience Replay
- Q(s,a) = V(s) + A(s,a) - mean(A)
- Target network for stable learning
- Epsilon-greedy exploration

### Tasks (`tasks.py`)
- **3 difficulty tiers**: Easy (5 stops), Medium (10 stops), Hard (12 stops)
- **5 task configurations**: task_1 through task_5
- Configurable parameters: fuel, demand, penalties, rewards

### Graders (`grader.py`)
- **5 grader functions**: `grade_task_1()` through `grade_task_5()`
- Deterministic evaluation against baselines
- Returns normalized score in [0.0, 1.0]
- Metrics: wait time, reward, fuel efficiency, coverage, balance

### Server (`server/app.py`)
- **FastAPI** server with OpenEnv endpoints
- `/reset`, `/step`, `/state` for environment interaction
- Dashboard with real-time visualization
- Gradio interface for interactive demos

## Validation & Testing

### Quick Validation
```bash
cd rl-bus-optimization
python tests/FINAL_CHECK.py
```

### Comprehensive Validation
```bash
python tests/final_validation.py
```

### Exact Validator Simulation
```bash
python tests/test_exact_validator_flow.py
```

## OpenEnv Compliance

### Required Components ✓
- [x] `openenv.yaml` with tasks and grading configuration
- [x] 5 tasks with graders (exceeds minimum of 3)
- [x] Grader functions return scores in [0.0, 1.0]
- [x] `inference.py` with structured logging
- [x] Docker container support
- [x] FastAPI server with OpenEnv endpoints

### Validation Status
- **Phase 1**: ✓ HF Space deploys
- **Phase 2**: ✓ 5 tasks with graders (>= 3 required)
- **Phase 3**: ✓ OpenEnv spec compliance
- **Phase 4**: ✓ Dockerfile builds
- **Phase 5**: ✓ Baseline reproduces

## Running the Project

### Training
```bash
python train.py --episodes 1000 --save-path models/dqn_bus.pt
```

### Evaluation
```bash
python grader.py --model-path models/dqn_bus_v6_best.pt
```

### Server
```bash
python server/app.py
# Access at http://localhost:7860
```

### Inference
```bash
python inference.py --task task_1 --mode dqn
```

## Key Features

1. **Real-World Data**: GTFS-calibrated demand from Indian cities
2. **Advanced RL**: Dueling DDQN + PER for sample efficiency
3. **Multi-Task**: 5 tasks across 3 difficulty levels
4. **OpenEnv Compliant**: Full specification compliance
5. **Production Ready**: Docker, FastAPI, comprehensive testing

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- OpenEnv-core 0.2.0+
- FastAPI, Gradio, Pydantic
- NumPy, Pandas, PyYAML

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub.
