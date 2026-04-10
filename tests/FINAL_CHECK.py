"""
FINAL CHECK - Simple validation without Unicode characters
Run this before submitting to verify everything works.
"""

import sys
import yaml
import importlib
import numpy as np


def main():
    print("="*70)
    print("FINAL PRE-SUBMISSION CHECK")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Load openenv.yaml
    print("\n[1/5] Loading openenv.yaml...")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        tasks = config.get("tasks", [])
        print(f"  PASS: Found {len(tasks)} tasks")
        
        if len(tasks) < 3:
            print(f"  FAIL: Need at least 3 tasks")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False
    
    # Test 2: Check grader module
    print("\n[2/5] Checking grader module...")
    try:
        import grader
        if hasattr(grader, "__all__"):
            print(f"  PASS: grader.__all__ exists")
        else:
            print(f"  FAIL: grader.__all__ missing")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False
    
    # Test 3: Check grader functions
    print("\n[3/5] Checking grader functions...")
    try:
        from grader import grade_task_1, grade_task_2, grade_task_3, grade_task_4, grade_task_5
        print(f"  PASS: All 5 grader functions imported")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False
    
    # Test 4: Resolve YAML grader paths
    print("\n[4/5] Resolving YAML grader paths...")
    try:
        tasks_with_graders = 0
        for task in config["tasks"]:
            grader_path = task.get("grader")
            if grader_path and ":" in grader_path:
                module_name, func_name = grader_path.split(":")
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                if callable(func):
                    tasks_with_graders += 1
        
        print(f"  PASS: {tasks_with_graders} tasks with valid graders")
        
        if tasks_with_graders < 3:
            print(f"  FAIL: Need at least 3 tasks with graders")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False
    
    # Test 5: Execute graders
    print("\n[5/5] Executing graders...")
    try:
        def test_policy(obs: np.ndarray) -> int:
            return 0
        
        from grader import grade_task_1, grade_task_2, grade_task_3
        
        scores = []
        for i, func in enumerate([grade_task_1, grade_task_2, grade_task_3], 1):
            score = func(test_policy, episodes=1)
            if isinstance(score, (float, int)) and 0.0 <= score <= 1.0:
                scores.append(score)
        
        print(f"  PASS: {len(scores)}/3 graders executed successfully")
        
        if len(scores) < 3:
            print(f"  FAIL: Not all graders executed")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False
    
    # Final verdict
    print("\n" + "="*70)
    if all_passed:
        print("SUCCESS: ALL CHECKS PASSED")
        print("\nYour submission is ready!")
        print("You will NOT get the 'Not enough tasks with graders' error.")
        print("\nNext steps:")
        print("  1. git add .")
        print("  2. git commit -m 'Fix: Expose grader functions'")
        print("  3. git push origin main")
        print("  4. Resubmit to hackathon")
    else:
        print("FAILURE: SOME CHECKS FAILED")
        print("\nPlease fix the errors above before submitting.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
