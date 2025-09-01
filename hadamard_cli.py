#!/usr/bin/env python3
"""
Simple CLI for Hadamard factorization testing.
"""

import numpy as np
import sys
from hadamard_real_demo import solve_hadamard_alternating
from hadamard_rigorous_theory import compute_alpha_data

def test_matrix(C):
    """Test if a matrix is factorizable."""
    print(f"\nTesting matrix:")
    print(C)
    print(f"\nRank: {np.linalg.matrix_rank(C)}")
    print(f"Zeros: {np.sum(np.abs(C) < 1e-10)}")
    
    # Try factorization
    best_error = float('inf')
    best_A, best_B = None, None
    
    for attempt in range(5):
        A, B, error, _ = solve_hadamard_alternating(C, max_iter=200)
        if error < best_error:
            best_error = error
            best_A, best_B = A, B
    
    rel_error = best_error / (np.linalg.norm(C, 'fro') + 1e-10)
    
    if rel_error < 1e-6:
        print(f"\n✅ FACTORIZABLE! Relative error: {rel_error:.2e}")
        print(f"Rank A: {np.linalg.matrix_rank(best_A)}, Rank B: {np.linalg.matrix_rank(best_B)}")
        
        # Check certification
        alpha_data = compute_alpha_data(best_A, best_B, C)
        if alpha_data['certified']:
            print(f"✅ Certified by α-criterion: α = {alpha_data['alpha']:.6f}")
        else:
            print(f"⚠️  Not certified: α = {alpha_data['alpha']:.6f}")
    else:
        print(f"\n❌ NOT FACTORIZABLE. Best error: {rel_error:.2e}")
    
    return rel_error < 1e-6

if __name__ == "__main__":
    # Test examples
    print("="*60)
    print("HADAMARD FACTORIZATION TESTS")
    print("="*60)
    
    # Example 1: Zero-free matrix
    print("\n1. Zero-free matrix (often factorizable over R):")
    C1 = np.array([[1.5, -0.3, 0.8, 1.2],
                   [-0.3, 2.1, 0.5, -0.7],
                   [0.8, 0.5, 1.8, 0.9],
                   [1.2, -0.7, 0.9, 1.3]])
    test_matrix(C1)
    
    # Example 2: Matrix with one zero row
    print("\n\n2. Matrix with zero row (should be factorizable):")
    C2 = np.array([[0, 0, 0, 0],
                   [1.2, -0.5, 0.8, 1.1],
                   [-0.3, 2.1, 0.5, -0.7],
                   [0.8, 0.5, 1.8, 0.9]])
    test_matrix(C2)
    
    # Example 3: Death valley pattern
    print("\n\n3. Death valley pattern (5 scattered zeros):")
    C3 = np.array([[0, 1.5, 0.8, 1.2],
                   [-0.3, 0, 0.5, -0.7],
                   [0.8, 0.5, 0, 0.9],
                   [1.2, 0, 0.9, 0]])
    test_matrix(C3)
    
    # Example 4: Special factorizable pattern
    print("\n\n4. Special pattern (full first row zeros):")
    C4 = np.array([[0, 0, 0, 0],
                   [1.5, -0.3, 0.8, 1.2],
                   [-0.3, 2.1, 0.5, -0.7],
                   [0.8, 0.5, 1.8, 0.9]])
    test_matrix(C4)