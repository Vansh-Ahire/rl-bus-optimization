"""
Comprehensive OpenEnv validation script.
Mimics the checks performed by the Meta PyTorch Hackathon validator.
"""

import sys
import importlib
import yaml
from typing import List, Tuple


def validate_grader_module() -> Tuple[bool, List[str]]:
    """Validate that grader module is properly structured."""
    errors = []
    
    try:
        grader = importlib.import_module("grader")
    except ImportError as e:
        errors.append(f"Cannot import grader module: {e}")
        return False, errors
    
    # Check __all__ exists
    if not hasattr(grader, "__all__"):
        errors.append("grader module missing __all__ export list")
    
    # Check for required grader functions
    required_graders = [
        "grade_task_1",
        "grade_task_2",
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    found = 0
    for grader_name in required_graders:
        if hasattr(grader, grader_name):
            func = getattr(grader, grader_name)
            if callable(func):
                found += 1
            else:
                errors.append(f"{grader_name} exists but is not callable")
        else:
            errors.append(f"{grader_name} not found in grader module")
    
    if found < 3:
        errors.append(f"Only {found} grader functions found (minimum 3 required)")
        return False, errors
    
    return True, errors


def validate_openenv_yaml() -> Tuple[bool, List[str]]:
    """Validate openenv.yaml structure and grader references."""
    errors = []
    
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Cannot load openenv.yaml: {e}")
        return False, errors
    
    # Check tasks section
    tasks = config.get("tasks", [])
    if len(tasks) < 3:
        errors.append(f"Only {len(tasks)} tasks defined (minimum 3 required)")
    
    # Check each task has a grader
    tasks_with_graders = 0
    for task in tasks:
        task_id = task.get("id")
        grader_path = task.get("grader")
        
        if not grader_path:
            errors.append(f"Task '{task_id}' missing grader field")
            continue
        
        # Try to resolve grader path
        try:
            module_name, func_name = grader_path.split(":")
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            if callable(func):
                tasks_with_graders += 1
            else:
                errors.append(f"Grader '{grader_path}' is not callable")
        except Exception as e:
            errors.append(f"Cannot resolve grader '{grader_path}': {e}")
    
    if tasks_with_graders < 3:
        errors.append(f"Only {tasks_with_graders} tasks with valid graders (minimum 3 required)")
        return False, errors
    
    # Check grading section
    grading = config.get("grading", {})
    if not grading:
        errors.append("Missing 'grading' section in openenv.yaml")
    
    per_task = grading.get("per_task", [])
    if len(per_task) < 3:
        errors.append(f"Only {len(per_task)} per-task graders in grading section (minimum 3 required)")
    
    return True, errors


def validate_grader_execution() -> Tuple[bool, List[str]]:
    """Test that graders can actually be executed."""
    errors = []
    
    try:
        import numpy as np
        from grader import grade_task_1
        
        def dummy_policy(obs: np.ndarray) -> int:
            return 0
        
        score = grade_task_1(dummy_policy, episodes=1)
        
        if not isinstance(score, float):
            errors.append(f"Grader returned {type(score)} instead of float")
            return False, errors
        
        if not (0.0 <= score <= 1.0):
            errors.append(f"Grader returned score {score} outside valid range [0.0, 1.0]")
            return False, errors
        
    except Exception as e:
        errors.append(f"Failed to execute grader: {e}")
        return False, errors
    
    return True, errors


def main():
    """Run all validation checks."""
    print("="*70)
    print("OpenEnv Validation Report")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Grader module structure
    print("\n[1/3] Validating grader module structure...")
    passed, errors = validate_grader_module()
    if passed:
        print("  ✓ PASS: Grader module properly structured")
    else:
        print("  ✗ FAIL: Grader module validation failed")
        all_passed = False
    
    for error in errors:
        print(f"    - {error}")
    
    # Test 2: openenv.yaml configuration
    print("\n[2/3] Validating openenv.yaml configuration...")
    passed, errors = validate_openenv_yaml()
    if passed:
        print("  ✓ PASS: openenv.yaml properly configured")
    else:
        print("  ✗ FAIL: openenv.yaml validation failed")
        all_passed = False
    
    for error in errors:
        print(f"    - {error}")
    
    # Test 3: Grader execution
    print("\n[3/3] Testing grader execution...")
    passed, errors = validate_grader_execution()
    if passed:
        print("  ✓ PASS: Graders execute successfully")
    else:
        print("  ✗ FAIL: Grader execution failed")
        all_passed = False
    
    for error in errors:
        print(f"    - {error}")
    
    # Final verdict
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("Your submission should pass Phase 2 validation!")
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please fix the errors above before resubmitting.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
