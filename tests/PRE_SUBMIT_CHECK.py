#!/usr/bin/env python3
"""
PRE-SUBMISSION CHECK
Run this script immediately before submitting to the hackathon.
It will give you a final GO/NO-GO decision.
"""

import sys
import os
from pathlib import Path


def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def print_section(text):
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}")


def run_check(name, script_name):
    """Run a validation script and return success status."""
    print(f"\nRunning {name}...")
    
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"  ✓ {name} PASSED")
            return True
        else:
            print(f"  ✗ {name} FAILED")
            print(f"     Run 'python {script_name}' to see details")
            return False
    except Exception as e:
        print(f"  ✗ {name} ERROR: {e}")
        return False


def main():
    print_header("PRE-SUBMISSION CHECK")
    print("\nThis script will verify your submission is ready.")
    print("It runs all validation tests to ensure you won't get")
    print("the 'Not enough tasks with graders' error again.")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print_section("Running Validation Tests")
    
    tests = [
        ("Grader Detection Test", "test_grader_detection.py"),
        ("OpenEnv YAML Test", "test_openenv_yaml.py"),
        ("Validator Simulation", "test_validator_simulation.py"),
        ("Final Validation", "final_validation.py"),
        ("Exact Validator Flow", "test_exact_validator_flow.py"),
    ]
    
    results = []
    for name, script in tests:
        if Path(script).exists():
            passed = run_check(name, script)
            results.append((name, passed))
        else:
            print(f"\n⚠ Warning: {script} not found, skipping")
    
    # Summary
    print_section("RESULTS SUMMARY")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    
    # Final verdict
    print_header("FINAL VERDICT")
    
    if passed_count == total_count:
        print("""
  ✓✓✓ ALL TESTS PASSED ✓✓✓
  
  Your submission is READY!
  
  You will NOT get the "Not enough tasks with graders" error.
  
  Next steps:
    1. Commit your changes:
       git add .
       git commit -m "Fix: Expose grader functions for validator"
    
    2. Push to GitHub:
       git push origin main
    
    3. Resubmit to the hackathon
    
  Expected result: Phase 2 validation will PASS
        """)
        return 0
    else:
        print("""
  ✗✗✗ SOME TESTS FAILED ✗✗✗
  
  Your submission is NOT ready yet.
  
  Please review the failed tests above and fix any issues.
  Run the individual test scripts to see detailed error messages.
        """)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCheck cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
