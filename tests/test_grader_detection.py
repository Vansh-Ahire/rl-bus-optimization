"""
Quick test to verify grader functions are properly exposed and callable.
This mimics what the OpenEnv validator does.
"""

import sys
import importlib


def test_grader_detection():
    """Test that all 5 grader functions can be discovered and called."""
    
    # Test 1: Import grader module
    try:
        grader = importlib.import_module("grader")
        print("✓ Successfully imported grader module")
    except ImportError as e:
        print(f"✗ Failed to import grader module: {e}")
        return False
    
    # Test 2: Check __all__ exports
    if hasattr(grader, "__all__"):
        print(f"✓ grader.__all__ exists: {grader.__all__}")
    else:
        print("✗ grader.__all__ not found")
        return False
    
    # Test 3: Verify all 5 grader functions exist
    expected_graders = [
        "grade_task_1",
        "grade_task_2", 
        "grade_task_3",
        "grade_task_4",
        "grade_task_5",
    ]
    
    found_graders = []
    for grader_name in expected_graders:
        if hasattr(grader, grader_name):
            func = getattr(grader, grader_name)
            if callable(func):
                found_graders.append(grader_name)
                print(f"✓ Found callable {grader_name}")
            else:
                print(f"✗ {grader_name} exists but is not callable")
        else:
            print(f"✗ {grader_name} not found in grader module")
    
    # Test 4: Check if we have at least 3 graders (OpenEnv requirement)
    if len(found_graders) >= 3:
        print(f"\n✓ PASS: Found {len(found_graders)} grader functions (minimum 3 required)")
    else:
        print(f"\n✗ FAIL: Only found {len(found_graders)} grader functions (minimum 3 required)")
        return False
    
    # Test 5: Test calling a grader with a simple policy
    try:
        import numpy as np
        
        def dummy_policy(obs: np.ndarray) -> int:
            """Simple random policy for testing."""
            return 0
        
        # Try calling grade_task_1 with minimal episodes
        score = grader.grade_task_1(dummy_policy, episodes=1)
        
        if isinstance(score, float) and 0.0 <= score <= 1.0:
            print(f"✓ grade_task_1 executed successfully, returned score: {score:.4f}")
        else:
            print(f"✗ grade_task_1 returned invalid score: {score}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to execute grade_task_1: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED - Graders should be detectable by OpenEnv")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_grader_detection()
    sys.exit(0 if success else 1)
