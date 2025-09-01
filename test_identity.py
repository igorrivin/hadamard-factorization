#!/usr/bin/env python3
"""
Test factorization of the identity matrix.
"""

import numpy as np
from hadamard_real_demo import solve_hadamard_alternating
from hadamard_real_certified import find_rank2_basis

# Identity matrix
I = np.eye(4)

print("Testing factorization of the identity matrix:")
print(I)
print(f"\nRank: {np.linalg.matrix_rank(I)}")
print(f"Zeros: {np.sum(np.abs(I) < 1e-10)}")

# Try factorization
best_error = float('inf')
best_A, best_B = None, None

for attempt in range(10):  # More attempts for this special case
    A, B, error, _ = solve_hadamard_alternating(I, max_iter=500)
    if error < best_error:
        best_error = error
        best_A, best_B = A, B

rel_error = best_error / (np.linalg.norm(I, 'fro') + 1e-10)

print(f"\nRelative error: {rel_error:.2e}")
print(f"Absolute error: {best_error:.2e}")

if rel_error < 1e-6:
    print("\n✅ FACTORIZABLE!")
    print(f"Rank A: {np.linalg.matrix_rank(best_A)}")
    print(f"Rank B: {np.linalg.matrix_rank(best_B)}")
    
    print("\nFactor A:")
    print(best_A)
    
    print("\nFactor B:")
    print(best_B)
    
    print("\nVerification: A ∘ B =")
    print(best_A * best_B)
    
    print("\nDifference from I:")
    print(best_A * best_B - I)
    
    # Find rank-2 decompositions
    print("\n" + "="*50)
    print("RANK-2 DECOMPOSITIONS")
    print("="*50)
    
    basis_A = find_rank2_basis(best_A)
    if basis_A is not None:
        U_A, V_A = basis_A
        print("\nA = U_A @ V_A^T where:")
        print("U_A =")
        print(U_A)
        print("\nV_A =")
        print(V_A)
        print("\nVerification: U_A @ V_A^T =")
        print(U_A @ V_A.T)
    
    basis_B = find_rank2_basis(best_B)
    if basis_B is not None:
        U_B, V_B = basis_B
        print("\nB = U_B @ V_B^T where:")
        print("U_B =")
        print(U_B)
        print("\nV_B =")
        print(V_B)
        print("\nVerification: U_B @ V_B^T =")
        print(U_B @ V_B.T)
    
    # Check for patterns
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    # Check if A and B have special structure
    print("\nChecking for diagonal structure...")
    A_diag = np.diag(np.diag(best_A))
    B_diag = np.diag(np.diag(best_B))
    
    print(f"A diagonal part: {np.diag(best_A)}")
    print(f"B diagonal part: {np.diag(best_B)}")
    print(f"Product of diagonals: {np.diag(best_A) * np.diag(best_B)}")
    
    # Check off-diagonal structure
    A_off = best_A - A_diag
    B_off = best_B - B_diag
    print(f"\n||A_off-diagonal||_F = {np.linalg.norm(A_off, 'fro'):.3f}")
    print(f"||B_off-diagonal||_F = {np.linalg.norm(B_off, 'fro'):.3f}")
    
else:
    print("\n❌ NOT FACTORIZABLE")
    print("This would be surprising since I has many zeros!")