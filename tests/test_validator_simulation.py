"""
Simulate the exact validation logic that the Meta PyTorch Hackathon validator uses.
This tests grader detection from multiple angles.
"""

import sys
import os
import yaml
import importlib
import importlib.util
from pathlib import Path


def test_method_1_direct_import():
    """Method 1: Direct module import (most common)"""
    print("\n[Method 1] Testing direct import...")
    try:
        import grader
        
        grader_functions = [
            "grade_task_1",
            "grade_task_2",
            "grade_task_3",
            "grade_task_4",
            "grade_task_5",
        ]
        
        found = 0
        for func_name in grader_functions:
            if hasattr(grader, func_name) and callable(getattr(grader, func_name)):
                found += 1
                print(f"  ✓ Found {func_name}")
        
        print(f"  Result: {found}/5 graders found")
        return found >= 3
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_method_2_yaml_resolution():
    """Method 2: Resolve graders from openenv.yaml paths"""
    print("\n[Method 2] Testing YAML path resolution...")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        tasks = config.get("tasks", [])
        found = 0
        
        for task in tasks:
            grader_path = task.get("grader")
            if not grader_path:
                continue
            
            try:
                module_name, func_name = grader_path.split(":")
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                
                if callable(func):
                    found += 1
                    print(f"  ✓ Resolved {grader_path}")
            except Exception as e:
                print(f"  ✗ Failed to resolve {grader_path}: {e}")
        
        print(f"  Result: {found}/5 graders resolved")
        return found >= 3
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_method_3_file_import():
    """Method 3: Import from file path (for validators that use file-based imports)"""
    print("\n[Method 3] Testing file-based import...")
    try:
        grader_path = Path("grader.py")
        if not grader_path.exists():
            print(f"  ✗ grader.py not found")
            return False
        
        spec = importlib.util.spec_from_file_location("grader", grader_path)
        grader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grader)
        
        grader_functions = [
            "grade_task_1",
            "grade_task_2",
            "grade_task_3",
            "grade_task_4",
            "grade_task_5",
        ]
        
        found = 0
        for func_name in grader_functions:
            if hasattr(grader, func_name) and callable(getattr(grader, func_name)):
                found += 1
                print(f"  ✓ Found {func_name}")
        
        print(f"  Result: {found}/5 graders found")
        return found >= 3
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_method_4_package_import():
    """Method 4: Import as package (if validator treats directory as package)"""
    print("\n[Method 4] Testing package import...")
    try:
        # Try importing from parent directory as package
        parent_dir = Path.cwd().parent
        sys.path.insert(0, str(parent_dir))
        
        package_name = Path.cwd().name
        grader_module = importlib.import_module(f"{package_name}.grader")
        
        grader_functions = [
            "grade_task_1",
            "grade_task_2",
            "grade_task_3",
            "grade_task_4",
            "grade_task_5",
        ]
        
        found = 0
        for func_name in grader_functions:
            if hasattr(grader_module, func_name) and callable(getattr(grader_module, func_name)):
                found += 1
                print(f"  ✓ Found {func_name}")
        
        print(f"  Result: {found}/5 graders found")
        return found >= 3
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_method_5_grading_section():
    """Method 5: Check grading section in openenv.yaml"""
    print("\n[Method 5] Testing grading section...")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        grading = config.get("grading", {})
        if not grading:
            print("  ✗ No grading section found")
            return False
        
        module_name = grading.get("module")
        if not module_name:
            print("  ✗ No module specified in grading section")
            return False
        
        print(f"  ✓ Grading module: {module_name}")
        
        per_task = grading.get("per_task", [])
        if len(per_task) < 3:
            print(f"  ✗ Only {len(per_task)} per_task entries (need >= 3)")
            return False
        
        print(f"  ✓ Found {len(per_task)} per_task entries")
        
        # Try to import the module and verify functions
        try:
            module = importlib.import_module(module_name)
            found = 0
            
            for entry in per_task:
                func_name = entry.get("function")
                if hasattr(module, func_name) and callable(getattr(module, func_name)):
                    found += 1
                    print(f"  ✓ Verified {func_name}")
            
            print(f"  Result: {found}/{len(per_task)} functions verified")
            return found >= 3
        except Exception as e:
            print(f"  ✗ Failed to verify functions: {e}")
            return False
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_method_6_execution():
    """Method 6: Actually execute a grader to ensure it works"""
    print("\n[Method 6] Testing grader execution...")
    try:
        import numpy as np
        from grader import grade_task_1, grade_task_2, grade_task_3
        
        def dummy_policy(obs: np.ndarray) -> int:
            return 0
        
        scores = []
        for i, grader_func in enumerate([grade_task_1, grade_task_2, grade_task_3], 1):
            try:
                score = grader_func(dummy_policy, episodes=1)
                if isinstance(score, float) and 0.0 <= score <= 1.0:
                    scores.append(score)
                    print(f"  ✓ grade_task_{i} executed: {score:.4f}")
                else:
                    print(f"  ✗ grade_task_{i} returned invalid score: {score}")
            except Exception as e:
                print(f"  ✗ grade_task_{i} failed: {e}")
        
        print(f"  Result: {len(scores)}/3 graders executed successfully")
        return len(scores) >= 3
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("="*70)
    print("COMPREHENSIVE VALIDATOR SIMULATION")
    print("Testing all possible grader detection methods")
    print("="*70)
    
    methods = [
        ("Direct Import", test_method_1_direct_import),
        ("YAML Path Resolution", test_method_2_yaml_resolution),
        ("File-Based Import", test_method_3_file_import),
        ("Package Import", test_method_4_package_import),
        ("Grading Section", test_method_5_grading_section),
        ("Execution Test", test_method_6_execution),
    ]
    
    results = []
    for name, test_func in methods:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*70)
    if passed_count == len(methods):
        print("✓ ALL METHODS PASSED - Graders should be detectable!")
    elif passed_count >= 4:
        print(f"⚠ {passed_count}/{len(methods)} methods passed - Should work but verify")
    else:
        print(f"✗ Only {passed_count}/{len(methods)} methods passed - May fail validation")
    print("="*70)
    
    return 0 if passed_count >= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
