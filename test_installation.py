#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed and basic functionality works.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'scikit-learn'),
        ('umap', 'UMAP'),
        ('pickle', 'Pickle (built-in)')
    ]
    
    all_good = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_good = False
    
    return all_good

def test_basic_functionality():
    """Test basic matrix operations."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        
        # Test matrix creation
        M = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1]])
        print("  ✓ Matrix creation")
        
        # Test rank computation
        rank = np.linalg.matrix_rank(M)
        assert rank == 2, f"Expected rank 2, got {rank}"
        print("  ✓ Rank computation")
        
        # Test Hadamard product
        A = np.array([[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]])
        C = M * A
        print("  ✓ Hadamard product")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_file_operations():
    """Test file I/O operations."""
    print("\nTesting file operations...")
    
    try:
        import pickle
        import numpy as np
        
        # Test pickle save/load
        test_data = [1, 2, 3, 4, 5]
        with open('test_temp.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        with open('test_temp.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        
        assert loaded_data == test_data
        print("  ✓ Pickle save/load")
        
        # Clean up
        import os
        os.remove('test_temp.pkl')
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("HADAMARD FACTORIZATION INSTALLATION TEST")
    print("="*50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\nSome packages are missing. Install with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    # Test file operations
    files_ok = test_file_operations()
    
    if imports_ok and basic_ok and files_ok:
        print("\n✓ All tests passed! Ready to run the main scripts.")
        print("\nSuggested order:")
        print("  1. python hadamard_search.py      # Run main search (~1 minute)")
        print("  2. python verify_integers.py      # Verify counterexamples")
        print("  3. python analyze_results.py      # Analyze patterns")
        print("  4. python visualize_matrices.py   # Create visualizations")
        print("\nOr run all at once with: ./run_tests.sh")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()