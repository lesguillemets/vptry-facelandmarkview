#!/usr/bin/env python3
"""
Simple test to verify imports work correctly after refactoring
"""

import sys

def test_package_imports():
    """Test that the package can be imported"""
    print("Testing package imports...")
    
    # Test that we can import from the package (without Qt, which requires display)
    try:
        from vptry_facelandmarkview import constants
        print(f"✓ constants module imported")
        print(f"  POINT_SIZE = {constants.POINT_SIZE}")
        print(f"  SCALE_MARGIN = {constants.SCALE_MARGIN}")
    except Exception as e:
        print(f"✗ Failed to import constants: {e}")
        return False
    
    # Test utils can be imported (but not instantiated without Qt)
    try:
        from vptry_facelandmarkview import utils
        print(f"✓ utils module imported")
        print(f"  Functions: filter_nan_landmarks, calculate_center_and_scale, draw_landmarks")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    # Test backward compatibility
    try:
        import facelandmarkview
        print(f"✓ Backward compatibility wrapper works")
        print(f"  Exports: {facelandmarkview.__all__}")
    except Exception as e:
        print(f"✗ Failed to import facelandmarkview wrapper: {e}")
        return False
    
    return True

def test_numpy_functions():
    """Test that numpy functions work"""
    print("\nTesting numpy utility functions...")
    try:
        import numpy as np
        from vptry_facelandmarkview.utils import filter_nan_landmarks, calculate_center_and_scale
        
        # Create test data with some NaN values
        test_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ])
        
        valid_landmarks, valid_mask = filter_nan_landmarks(test_data)
        print(f"✓ filter_nan_landmarks works")
        print(f"  Input shape: {test_data.shape}")
        print(f"  Valid landmarks: {len(valid_landmarks)}")
        print(f"  Expected: 2 valid landmarks (row with NaN filtered out)")
        
        assert len(valid_landmarks) == 2, "Should have 2 valid landmarks"
        
        # Test calculate_center_and_scale
        center, scale = calculate_center_and_scale(valid_landmarks)
        print(f"✓ calculate_center_and_scale works")
        print(f"  Center: {center}")
        print(f"  Scale: {scale}")
        
        return True
    except Exception as e:
        print(f"✗ NumPy functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Refactored Package Structure")
    print("=" * 60)
    print()
    
    success = True
    
    if not test_package_imports():
        success = False
    
    if not test_numpy_functions():
        success = False
    
    print()
    print("=" * 60)
    if success:
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    else:
        print("Some tests failed!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
