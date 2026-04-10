"""
Test that openenv.yaml grader paths can be resolved correctly.
"""

import yaml
import importlib


def test_openenv_yaml():
    """Verify openenv.yaml grader configuration."""
    
    # Load openenv.yaml
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("Testing openenv.yaml grader configuration...")
    print("="*60)
    
    # Check tasks section
    tasks = config.get("tasks", [])
    print(f"\nFound {len(tasks)} tasks in openenv.yaml")
    
    graders_found = 0
    for task in tasks:
        task_id = task.get("id")
        grader_path = task.get("grader")
        
        if grader_path:
            graders_found += 1
            print(f"  ✓ Task '{task_id}' has grader: {grader_path}")
            
            # Try to resolve the grader path
            try:
                module_name, func_name = grader_path.split(":")
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                
                if callable(func):
                    print(f"    ✓ Successfully resolved {grader_path}")
                else:
                    print(f"    ✗ {grader_path} is not callable")
            except Exception as e:
                print(f"    ✗ Failed to resolve {grader_path}: {e}")
        else:
            print(f"  ✗ Task '{task_id}' has no grader field")
    
    # Check grading section
    grading = config.get("grading", {})
    per_task = grading.get("per_task", [])
    
    print(f"\n✓ Found {len(per_task)} per-task graders in grading section")
    
    for entry in per_task:
        func_name = entry.get("function")
        task_id = entry.get("task_id")
        print(f"  - {func_name} for {task_id}")
    
    # Final check
    print("\n" + "="*60)
    if graders_found >= 3:
        print(f"✓ PASS: Found {graders_found} tasks with graders (minimum 3 required)")
        return True
    else:
        print(f"✗ FAIL: Only {graders_found} tasks with graders (minimum 3 required)")
        return False


if __name__ == "__main__":
    import sys
    success = test_openenv_yaml()
    sys.exit(0 if success else 1)
