#!/usr/bin/env python3
"""
Test factorization of permutation matrices.
"""

import numpy as np
from hadamard_real_demo import solve_hadamard_alternating

def test_matrix(C, name):
    """Test if a matrix is factorizable."""
    print(f"\n{name}:")
    print(C)
    print(f"Zeros: {np.sum(np.abs(C) < 1e-10)}")
    
    # Try factorization
    best_error = float('inf')
    for attempt in range(5):
        A, B, error, _ = solve_hadamard_alternating(C, max_iter=200)
        if error < best_error:
            best_error = error
    
    rel_error = best_error / (np.linalg.norm(C, 'fro') + 1e-10)
    
    if rel_error < 1e-6:
        print(f"✅ FACTORIZABLE! Error: {rel_error:.2e}")
    else:
        print(f"❌ NOT FACTORIZABLE. Error: {rel_error:.2e}")
    
    return rel_error < 1e-6

print("TESTING PERMUTATION MATRICES")
print("="*50)

# Identity (already tested)
I = np.eye(4)
test_matrix(I, "Identity")

# Simple transposition (1,2)
P12 = np.array([[0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)
test_matrix(P12, "Transposition (1,2)")

# Cyclic permutation
P_cycle = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0]], dtype=float)
test_matrix(P_cycle, "Cyclic permutation")

# Full permutation
P_full = np.array([[0, 0, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 0]], dtype=float)
test_matrix(P_full, "General permutation")

# Anti-diagonal
P_anti = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=float)
test_matrix(P_anti, "Anti-diagonal")

print("\n\nANALYSIS:")
print("-"*30)
print("Permutation matrices have exactly one 1 in each row and column.")
print("This means they have 12 zeros (like the identity) but different patterns.")
print("The factorizability depends on the specific permutation structure.")