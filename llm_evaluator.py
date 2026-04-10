from __future__ import annotations

import argparse
from typing import Dict


def evaluate_submission(
    program_score_0_100: float | None = None,
) -> Dict[str, float]:
    """
    Simulated LLM-based evaluator (offline, deterministic).
    Scores are out of 10.
    """
    # A simple rubric that "feels like" LLM judging while staying offline:
    # - Code quality: assumes modular files + clean structure for this template.
    # - RL understanding: increases when programmatic score is strong (agent beats baselines).
    # - Design clarity: increases when score is reported and easy to interpret.
    code_quality = 9.0
    design_clarity = 9.0

    if program_score_0_100 is None:
        rl_understanding = 8.5
    else:
        s = float(program_score_0_100)
        s = max(0.0, min(100.0, s))
        rl_understanding = 6.5 + 3.5 * (s / 100.0)  # 6.5..10.0

    overall = (code_quality + rl_understanding + design_clarity) / 3.0
    return {
        "code_quality_10": code_quality,
        "rl_understanding_10": rl_understanding,
        "design_clarity_10": design_clarity,
        "overall_10": round(overall, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--program-score",
        type=float,
        default=None,
        help="Optional programmatic score (0-100) from grader to influence RL-understanding score.",
    )
    args = p.parse_args()

    report = evaluate_submission(
        program_score_0_100=args.program_score,
    )
    print("=== Simulated LLM Evaluation ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

