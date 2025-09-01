#!/usr/bin/env python3
"""
Verify the MathOverflow factorization claim.
"""

import numpy as np

print("VERIFYING MATHOVERFLOW FACTORIZATION")
print("="*60)

# The claim from MathOverflow
A = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

B = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
], dtype=float)

C = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=float)

print("Matrix A (identity):")
print(A)
print(f"Rank(A) = {np.linalg.matrix_rank(A)}")

print("\nMatrix B:")
print(B)
print(f"Rank(B) = {np.linalg.matrix_rank(B)}")

print("\nMatrix C:")
print(C)
print(f"Rank(C) = {np.linalg.matrix_rank(C)}")

print("\nB ⊙ C =")
result = B * C  # Hadamard product
print(result)

print(f"\nDoes B ⊙ C = A? {np.allclose(result, A)}")

print("\n\nANALYSIS:")
print("-"*40)
print("The MathOverflow claim is CORRECT!")
print("The identity CAN be written as B ⊙ C where rank(B) = rank(C) = 2")

print("\nWhy this works:")
print("- B has block structure with rank 2")
print("- C has alternating column pattern with rank 2")
print("- Their Hadamard product cleverly produces the identity")

print("\nKey insight: The sparsity patterns of B and C are complementary:")
print("- Where we need 1's (diagonal), both B and C have 1's")
print("- Where we need 0's (off-diagonal), at least one of B or C has 0")

# Let's verify the structure more carefully
print("\n\nDETAILED VERIFICATION:")
print("-"*40)

for i in range(4):
    for j in range(4):
        print(f"Position ({i},{j}): B[{i},{j}]={B[i,j]:.0f} × C[{i},{j}]={C[i,j]:.0f} = {B[i,j]*C[i,j]:.0f}")
        
print("\nThis factorization is valid!")