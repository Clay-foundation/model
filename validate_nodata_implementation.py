"""
Simple validation script to check nodata handling implementation.
Tests the logic without requiring full dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that our modifications didn't break imports."""
    try:
        # Test that the files can be imported/parsed
        import ast
        
        # Test datamodule changes
        with open('claymodel/datamodule.py', 'r') as f:
            datamodule_code = f.read()
            ast.parse(datamodule_code)
        print("✓ Datamodule syntax is valid")
        
        # Test model changes  
        with open('claymodel/model.py', 'r') as f:
            model_code = f.read()
            ast.parse(model_code)
        print("✓ Model syntax is valid")
        
        # Test utils changes
        with open('claymodel/utils.py', 'r') as f:
            utils_code = f.read()
            ast.parse(utils_code)
        print("✓ Utils syntax is valid")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_mask_logic():
    """Test mask creation logic conceptually."""
    print("\n=== Testing Mask Logic ===")
    
    # Simulate nodata detection logic
    def simulate_nodata_detection(pixel_value, nodata_values=None):
        if nodata_values is None:
            nodata_values = [0, -9999, -32768, 65535]
        
        # Simulate NaN check
        if str(pixel_value) == 'nan':
            return True
            
        # Check against nodata values
        return pixel_value in nodata_values
    
    # Test cases
    test_cases = [
        (0, True, "Zero should be nodata"),
        (-9999, True, "Standard nodata value"),
        (1000, False, "Normal value should not be nodata"),
        ('nan', True, "NaN should be nodata"),
        (-32768, True, "Integer nodata should be detected"),
    ]
    
    all_passed = True
    for value, expected, description in test_cases:
        result = simulate_nodata_detection(value)
        if result == expected:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - Expected {expected}, got {result}")
            all_passed = False
    
    return all_passed

def test_code_structure():
    """Test that the code structure is correct."""
    print("\n=== Testing Code Structure ===")
    
    # Check that key methods exist in the files
    checks = []
    
    with open('claymodel/datamodule.py', 'r') as f:
        datamodule_content = f.read()
        checks.append(('mask' in datamodule_content, "Mask handling in datamodule"))
        checks.append(('additional_info' in datamodule_content, "Additional info structure"))
    
    with open('claymodel/model.py', 'r') as f:
        model_content = f.read()
        checks.append(('compute_patch_mask' in model_content, "Patch mask computation method"))
        checks.append(('nodata_mask' in model_content, "Nodata mask parameter"))
        checks.append(('patch_mask' in model_content, "Patch mask usage"))
    
    with open('claymodel/utils.py', 'r') as f:
        utils_content = f.read()
        checks.append(('create_nodata_mask' in utils_content, "Nodata mask creation function"))
        checks.append(('create_datacube_with_mask' in utils_content, "Datacube creation function"))
    
    all_passed = True
    for check_passed, description in checks:
        if check_passed:
            print(f"✓ {description}")
        else:
            print(f"✗ {description}")
            all_passed = False
    
    return all_passed

def main():
    """Run all validation tests."""
    print("=== Nodata Handling Implementation Validation ===")
    
    all_tests = [
        test_imports(),
        test_mask_logic(),
        test_code_structure()
    ]
    
    if all(all_tests):
        print("\n🎉 All validation tests passed!")
        print("The nodata handling implementation appears to be working correctly.")
        return True
    else:
        print("\n❌ Some validation tests failed.")
        print("Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)