"""
Simulate the EXACT flow the Meta PyTorch Hackathon validator uses.
Based on the validation requirements:
"Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range"
"""

import sys
import yaml
import importlib
import numpy as np


def simulate_validator():
    """
    Simulate the exact validator flow:
    1. Load openenv.yaml
    2. Enumerate tasks
    3. For each task with a grader:
       - Resolve the grader path (module:function)
       - Create a test policy
       - Run the grader
       - Verify score is in [0.0, 1.0]
    """
    
    print("="*70)
    print("SIMULATING META PYTORCH HACKATHON VALIDATOR")
    print("="*70)
    
    # Step 1: Load openenv.yaml
    print("\n[Step 1] Loading openenv.yaml...")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Loaded openenv.yaml")
    except Exception as e:
        print(f"  ✗ Failed to load openenv.yaml: {e}")
        return False
    
    # Step 2: Enumerate tasks
    print("\n[Step 2] Enumerating tasks...")
    tasks = config.get("tasks", [])
    print(f"  Found {len(tasks)} tasks")
    
    if len(tasks) < 3:
        print(f"  ✗ FAIL: Need at least 3 tasks, found {len(tasks)}")
        return False
    
    # Step 3: Check each task for grader
    print("\n[Step 3] Checking tasks for graders...")
    tasks_with_graders = []
    
    for task in tasks:
        task_id = task.get("id")
        grader_path = task.get("grader")
        
        if grader_path:
            tasks_with_graders.append((task_id, grader_path))
            print(f"  ✓ Task '{task_id}' has grader: {grader_path}")
        else:
            print(f"  ⚠ Task '{task_id}' has no grader")
    
    print(f"\n  Total tasks with graders: {len(tasks_with_graders)}")
    
    if len(tasks_with_graders) < 3:
        print(f"  ✗ FAIL: Need at least 3 tasks with graders, found {len(tasks_with_graders)}")
        return False
    
    print(f"  ✓ PASS: Found {len(tasks_with_graders)} tasks with graders (>= 3 required)")
    
    # Step 4: Run each grader
    print("\n[Step 4] Running graders...")
    
    # Create a simple test policy
    def test_policy(obs: np.ndarray) -> int:
        """Simple policy for testing - always returns action 0."""
        return 0
    
    successful_graders = 0
    failed_graders = []
    
    for task_id, grader_path in tasks_with_graders:
        print(f"\n  Testing {task_id} with grader {grader_path}...")
        
        try:
            # Parse module:function
            if ":" not in grader_path:
                raise ValueError(f"Invalid grader path format: {grader_path}")
            
            module_name, func_name = grader_path.split(":", 1)
            
            # Import module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(f"Cannot import module '{module_name}': {e}")
            
            # Get function
            if not hasattr(module, func_name):
                raise AttributeError(f"Module '{module_name}' has no function '{func_name}'")
            
            grader_func = getattr(module, func_name)
            
            if not callable(grader_func):
                raise TypeError(f"{grader_path} is not callable")
            
            # Run grader with test policy (minimal episodes for speed)
            print(f"    Executing {func_name}...")
            score = grader_func(test_policy, episodes=1)
            
            # Verify score type
            if not isinstance(score, (float, int)):
                raise TypeError(f"Grader returned {type(score)}, expected float")
            
            score = float(score)
            
            # Verify score range
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score {score} outside valid range [0.0, 1.0]")
            
            print(f"    ✓ SUCCESS: Score = {score:.4f} (valid range)")
            successful_graders += 1
            
        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            failed_graders.append((task_id, str(e)))
    
    # Step 5: Final verdict
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Tasks found: {len(tasks)}")
    print(f"Tasks with graders: {len(tasks_with_graders)}")
    print(f"Graders executed successfully: {successful_graders}")
    print(f"Graders failed: {len(failed_graders)}")
    
    if failed_graders:
        print("\nFailed graders:")
        for task_id, error in failed_graders:
            print(f"  - {task_id}: {error}")
    
    print("\n" + "="*70)
    
    # Validator passes if:
    # 1. At least 3 tasks with graders exist
    # 2. All graders execute successfully
    # 3. All scores are in [0.0, 1.0]
    
    if len(tasks_with_graders) < 3:
        print("✗ VALIDATION FAILED: Not enough tasks with graders")
        print(f"   Required: >= 3, Found: {len(tasks_with_graders)}")
        return False
    
    if successful_graders < 3:
        print("✗ VALIDATION FAILED: Not enough graders executed successfully")
        print(f"   Required: >= 3, Successful: {successful_graders}")
        return False
    
    if failed_graders:
        print("✗ VALIDATION FAILED: Some graders failed to execute")
        return False
    
    print("✓✓✓ VALIDATION PASSED ✓✓✓")
    print(f"\nYour submission meets the Phase 2 requirement:")
    print(f"  • {len(tasks_with_graders)} tasks with graders (>= 3 required)")
    print(f"  • All graders execute successfully")
    print(f"  • All scores in valid range [0.0, 1.0]")
    print("\n" + "="*70)
    
    return True


if __name__ == "__main__":
    success = simulate_validator()
    sys.exit(0 if success else 1)
