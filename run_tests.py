"""
run_tests.py
============
Automated Test Runner for the Sales & Revenue Forecasting Project.
Executes unit tests across data processing, modeling, inventory optimization,
and forecast evaluation.
"""

import sys
import pytest

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SUITE OF VERIFICATION TESTS FOR SALES & REVENUE FORECASTING")
    print("=" * 70)
    
    args = ["-v", "--tb=short", "tests"]
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"\nTEST SUITE FAILED WITH EXIT CODE: {exit_code}")
        
    sys.exit(exit_code)
