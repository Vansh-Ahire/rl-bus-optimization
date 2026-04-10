"""
Final comprehensive validation before submission.
This checks EVERYTHING that could possibly cause validation failure.
"""

import sys
import os
import yaml
import importlib
import inspect
from pathlib import Path
from typing import Callable
import numpy as np


class ValidationError(Exception):
    pass


def check_file_structure():
    """Check that all required files exist."""
    print("\n[1/10] Checking file structure...")
    
    required_files = [
        "openenv.yaml",
        "grader.py",
        "tasks.py",
        "environment.py",
        "__init__.py",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"  ✗ Missing: {file}")
        else:
            print(f"  ✓ Found: {file}")
    
    if missing:
        raise ValidationError(f"Missing required files: {missing}")
    
    print("  ✓ All required files present")


def check_openenv_yaml_structure():
    """Check openenv.yaml has correct structure."""
    print("\n[2/10] Checking openenv.yaml structure...")
    
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Check required top-level keys
    required_keys = ["name", "version", "tasks", "grading"]
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"openenv.yaml missing required key: {key}")
        print(f"  ✓ Has '{key}' section")
    
    # Check tasks
    tasks = config["tasks"]
    if not isinstance(tasks, list):
        raise ValidationError("tasks must be a list")
    
    if len(tasks) < 3:
        raise ValidationError(f"Need at least 3 tasks, found {len(tasks)}")
    
    print(f"  ✓ Has {len(tasks)} tasks (>= 3 required)")
    
    # Check each task has required fields
    for i, task in enumerate(tasks):
        required_task_fields = ["id", "name", "grader"]
        for field in required_task_fields:
            if field not in task:
                raise ValidationError(f"Task {i} missing field: {field}")
        
        # Check grader format
        grader = task["grader"]
        if ":" not in grader:
            raise ValidationError(f"Task {i} grader must be in format 'module:function', got: {grader}")
        
        print(f"  ✓ Task '{task['id']}' has grader: {grader}")
    
    # Check grading section
    grading = config["grading"]
    if "module" not in grading:
        raise ValidationError("grading section missing 'module' field")
    
    if "per_task" not in grading:
        raise ValidationError("grading section missing 'per_task' field")
    
    per_task = grading["per_task"]
    if len(per_task) < 3:
        raise ValidationError(f"grading.per_task needs >= 3 entries, found {len(per_task)}")
    
    print(f"  ✓ Grading section has {len(per_task)} per_task entries")
    
    # Verify consistency between tasks and per_task
    task_ids = {task["id"] for task in tasks}
    per_task_ids = {entry["task_id"] for entry in per_task}
    
    if not per_task_ids.issubset(task_ids):
        missing = per_task_ids - task_ids
        raise ValidationError(f"per_task references non-existent task_ids: {missing}")
    
    print("  ✓ Task IDs consistent between tasks and grading sections")


def check_grader_module_imports():
    """Check that grader module can be imported."""
    print("\n[3/10] Checking grader module imports...")
    
    try:
        import grader
        print("  ✓ Successfully imported grader module")
    except ImportError as e:
        raise ValidationError(f"Cannot import grader module: {e}")
    
    # Check __all__ exists
    if not hasattr(grader, "__all__"):
        raise ValidationError("grader module missing __all__ attribute")
    
    print(f"  ✓ grader.__all__ exists with {len(grader.__all__)} exports")
    
    # Check required functions in __all__
    required_graders = [
        "grade_task_1",
        "grade_task_2",
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    for func_name in required_graders:
        if func_name not in grader.__all__:
            raise ValidationError(f"{func_name} not in grader.__all__")
        print(f"  ✓ {func_name} in __all__")


def check_grader_functions_exist():
    """Check that all grader functions exist and are callable."""
    print("\n[4/10] Checking grader functions exist...")
    
    import grader
    
    required_graders = [
        "grade_task_1",
        "grade_task_2",
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    for func_name in required_graders:
        if not hasattr(grader, func_name):
            raise ValidationError(f"grader module missing function: {func_name}")
        
        func = getattr(grader, func_name)
        if not callable(func):
            raise ValidationError(f"{func_name} exists but is not callable")
        
        print(f"  ✓ {func_name} exists and is callable")


def check_grader_signatures():
    """Check that grader functions have correct signatures."""
    print("\n[5/10] Checking grader function signatures...")
    
    import grader
    
    required_graders = [
        "grade_task_1",
        "grade_task_2",
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    for func_name in required_graders:
        func = getattr(grader, func_name)
        sig = inspect.signature(func)
        
        # Check parameters
        params = list(sig.parameters.keys())
        if len(params) < 1:
            raise ValidationError(f"{func_name} must have at least 1 parameter")
        
        # First param should be agent_policy
        first_param = params[0]
        if first_param != "agent_policy":
            print(f"  ⚠ Warning: {func_name} first param is '{first_param}', expected 'agent_policy'")
        
        # Check for episodes parameter with default
        if "episodes" in params:
            episodes_param = sig.parameters["episodes"]
            if episodes_param.default == inspect.Parameter.empty:
                print(f"  ⚠ Warning: {func_name} 'episodes' parameter has no default value")
        
        # Check return annotation
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != float and str(sig.return_annotation) != 'float':
                print(f"  ⚠ Warning: {func_name} return type is {sig.return_annotation}, expected float")
        
        print(f"  ✓ {func_name} signature: {sig}")


def check_grader_docstrings():
    """Check that grader functions have docstrings."""
    print("\n[6/10] Checking grader function docstrings...")
    
    import grader
    
    required_graders = [
        "grade_task_1",
        "grade_task_2",
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    for func_name in required_graders:
        func = getattr(grader, func_name)
        if not func.__doc__:
            print(f"  ⚠ Warning: {func_name} has no docstring")
        else:
            print(f"  ✓ {func_name} has docstring")


def check_yaml_grader_resolution():
    """Check that all grader paths in YAML can be resolved."""
    print("\n[7/10] Checking YAML grader path resolution...")
    
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tasks = config["tasks"]
    
    for task in tasks:
        grader_path = task["grader"]
        module_name, func_name = grader_path.split(":")
        
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            if not callable(func):
                raise ValidationError(f"{grader_path} is not callable")
            
            print(f"  ✓ Resolved {grader_path}")
        except Exception as e:
            raise ValidationError(f"Cannot resolve {grader_path}: {e}")


def check_grader_execution():
    """Check that graders can actually execute."""
    print("\n[8/10] Checking grader execution...")
    
    from grader import grade_task_1, grade_task_2, grade_task_3
    
    def dummy_policy(obs: np.ndarray) -> int:
        """Simple test policy."""
        return 0
    
    test_graders = [
        ("grade_task_1", grade_task_1),
        ("grade_task_2", grade_task_2),
        ("grade_task_3", grade_task_3),
    ]
    
    for name, grader_func in test_graders:
        try:
            score = grader_func(dummy_policy, episodes=1)
            
            if not isinstance(score, float):
                raise ValidationError(f"{name} returned {type(score)}, expected float")
            
            if not (0.0 <= score <= 1.0):
                raise ValidationError(f"{name} returned {score}, must be in [0.0, 1.0]")
            
            print(f"  ✓ {name} executed successfully: {score:.4f}")
        except Exception as e:
            raise ValidationError(f"{name} execution failed: {e}")


def check_tasks_module():
    """Check that tasks module is properly configured."""
    print("\n[9/10] Checking tasks module...")
    
    try:
        from tasks import TASKS
        print(f"  ✓ Imported TASKS dictionary")
    except ImportError as e:
        raise ValidationError(f"Cannot import TASKS from tasks module: {e}")
    
    if not isinstance(TASKS, dict):
        raise ValidationError("TASKS must be a dictionary")
    
    if len(TASKS) < 3:
        raise ValidationError(f"TASKS must have at least 3 entries, found {len(TASKS)}")
    
    print(f"  ✓ TASKS has {len(TASKS)} task configurations")
    
    # Check that task IDs match openenv.yaml
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    yaml_task_ids = {task["id"] for task in config["tasks"]}
    tasks_keys = set(TASKS.keys())
    
    if yaml_task_ids != tasks_keys:
        missing_in_tasks = yaml_task_ids - tasks_keys
        missing_in_yaml = tasks_keys - yaml_task_ids
        if missing_in_tasks:
            raise ValidationError(f"TASKS missing task IDs from YAML: {missing_in_tasks}")
        if missing_in_yaml:
            print(f"  ⚠ Warning: TASKS has extra task IDs not in YAML: {missing_in_yaml}")
    
    print("  ✓ Task IDs consistent between YAML and tasks.py")


def check_package_init():
    """Check that __init__.py properly exposes graders."""
    print("\n[10/10] Checking __init__.py...")
    
    with open("__init__.py", "r") as f:
        init_content = f.read()
    
    # Check that grader functions are imported
    required_imports = [
        "grade_task_1",
        "grade_task_2",
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    for func_name in required_imports:
        if func_name not in init_content:
            print(f"  ⚠ Warning: {func_name} not found in __init__.py")
        else:
            print(f"  ✓ {func_name} imported in __init__.py")
    
    # Check __all__ in __init__.py
    if "__all__" not in init_content:
        print("  ⚠ Warning: __init__.py missing __all__")
    else:
        print("  ✓ __init__.py has __all__")


def main():
    print("="*70)
    print("FINAL COMPREHENSIVE VALIDATION")
    print("="*70)
    
    checks = [
        check_file_structure,
        check_openenv_yaml_structure,
        check_grader_module_imports,
        check_grader_functions_exist,
        check_grader_signatures,
        check_grader_docstrings,
        check_yaml_grader_resolution,
        check_grader_execution,
        check_tasks_module,
        check_package_init,
    ]
    
    failed = False
    for check in checks:
        try:
            check()
        except ValidationError as e:
            print(f"\n  ✗ VALIDATION FAILED: {e}")
            failed = True
            break
        except Exception as e:
            print(f"\n  ✗ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed = True
            break
    
    print("\n" + "="*70)
    if not failed:
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("\nYour submission is ready!")
        print("The graders are properly configured and should pass validation.")
        print("\nNext steps:")
        print("1. Commit all changes")
        print("2. Push to GitHub")
        print("3. Resubmit to the hackathon")
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
        print("\nPlease fix the errors above before submitting.")
    print("="*70)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
