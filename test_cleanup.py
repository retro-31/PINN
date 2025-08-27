#!/usr/bin/env python3
"""
Test script to verify that 2D Lagrange methods have been properly removed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test that the 2D Lagrange config files have been removed
def test_config_files_removed():
    config_files = [
        'configs/steady_state_2d_scalar_lagrange.py',
        'configs/steady_state_2d_vector_lagrange.py', 
        'configs/test_nonhomogeneous_lagrange.py'
    ]
    
    removed_files = []
    for config_file in config_files:
        if not os.path.exists(config_file):
            removed_files.append(config_file)
    
    if len(removed_files) == len(config_files):
        print("✅ All 2D Lagrange config files have been removed")
        return True
    else:
        print("❌ Some 2D Lagrange config files still exist:")
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"   - {config_file} still exists")
        return False

# Test that trying to create a 2D model with Lagrange method fails
def test_2d_lagrange_error():
    try:
        # Create a minimal config object
        class MockConfig:
            BC_METHOD = "lagrange"
            PROBLEM_TYPE = "steady_state_2d"  # This should trigger an error
            FIELD_TYPE = "scalar"
            DEVICE = "cpu"
            NUM_LAGRANGE_MULTIPLIERS = 10
        
        from pinn_lib.models import PINN
        
        try:
            # This should raise an error
            model = PINN([2, 10, 1], MockConfig())
            print("❌ 2D Lagrange model creation did not raise an error")
            return False
        except ValueError as e:
            if "Lagrange multiplier method only supported for 1D steady-state problems" in str(e):
                print("✅ Correct error raised for 2D Lagrange model creation")
                return True
            else:
                print(f"❌ Unexpected error for 2D Lagrange model: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Unexpected error during test: {e}")
        return False

# Test that 1D Lagrange still works
def test_1d_lagrange_works():
    try:
        class MockConfig:
            BC_METHOD = "lagrange"
            PROBLEM_TYPE = "steady_state_1d"  # This should work
            FIELD_TYPE = "scalar"
            DEVICE = "cpu"
            NUM_LAGRANGE_MULTIPLIERS = 2
        
        from pinn_lib.models import PINN
        
        model = PINN([1, 10, 1], MockConfig())
        if hasattr(model, 'lagrange_multipliers'):
            print("✅ 1D Lagrange model creation works correctly")
            return True
        else:
            print("❌ 1D Lagrange model missing lagrange_multipliers attribute")
            return False
            
    except Exception as e:
        print(f"❌ Error creating 1D Lagrange model: {e}")
        return False

if __name__ == "__main__":
    print("Testing cleanup of 2D Lagrange implementations...")
    print("=" * 60)
    
    results = []
    results.append(test_config_files_removed())
    results.append(test_2d_lagrange_error())
    results.append(test_1d_lagrange_works())
    
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 All tests passed! 2D Lagrange cleanup successful.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Cleanup may be incomplete.")
        sys.exit(1)
