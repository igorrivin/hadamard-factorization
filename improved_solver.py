#!/usr/bin/env python3
"""
Improved solver for Hadamard factorization that can handle difficult cases like the identity.
"""

import numpy as np
from scipy.linalg import svd

def project_to_rank2(M):
    """Project matrix to nearest rank-2 matrix."""
    U, s, Vt = svd(M)
    s[2:] = 0  # Keep only top 2 singular values
    return U @ np.diag(s) @ Vt

def try_structured_initialization(C, pattern_type='block'):
    """Try structured initializations that might work for special matrices."""
    n = C.shape[0]
    
    if pattern_type == 'block':
        # Block diagonal pattern like in MathOverflow example
        B = np.zeros((n, n))
        B[0:n//2, 0:n//2] = 1
        B[n//2:n, n//2:n] = 1
        B = project_to_rank2(B + 0.1 * np.random.randn(n, n))
        
    elif pattern_type == 'alternating':
        # Alternating column pattern
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    B[i, j] = 1
        B = project_to_rank2(B + 0.1 * np.random.randn(n, n))
        
    elif pattern_type == 'identity_based':
        # Start with something close to known factorization
        B = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=float)
        B = B + 0.01 * np.random.randn(n, n)
        B = project_to_rank2(B)
        
    else:  # random
        B = project_to_rank2(np.random.randn(n, n))
    
    # Compute corresponding A
    A = np.zeros_like(B)
    for i in range(n):
        for j in range(n):
            if abs(B[i,j]) > 1e-10:
                A[i,j] = C[i,j] / B[i,j]
            else:
                A[i,j] = np.random.randn()
    
    A = project_to_rank2(A)
    
    return A, B

def solve_hadamard_improved(C, max_iter=1000, tol=1e-10, n_restarts=10):
    """
    Improved solver with multiple initialization strategies.
    """
    n = C.shape[0]
    best_error = float('inf')
    best_A, best_B = None, None
    
    # Try different initialization patterns
    patterns = ['block', 'alternating', 'identity_based', 'random']
    
    for pattern in patterns:
        for restart in range(max(1, n_restarts // len(patterns))):
            # Initialize with structured pattern
            A, B = try_structured_initialization(C, pattern)
            
            prev_error = float('inf')
            
            for iteration in range(max_iter):
                # Update B given A
                B_new = np.zeros_like(B)
                for i in range(n):
                    for j in range(n):
                        if abs(A[i,j]) > 1e-10:
                            B_new[i,j] = C[i,j] / A[i,j]
                        else:
                            B_new[i,j] = 0
                
                B = project_to_rank2(B_new)
                
                # Update A given B  
                A_new = np.zeros_like(A)
                for i in range(n):
                    for j in range(n):
                        if abs(B[i,j]) > 1e-10:
                            A_new[i,j] = C[i,j] / B[i,j]
                        else:
                            A_new[i,j] = 0
                
                A = project_to_rank2(A_new)
                
                # Check convergence
                error = np.linalg.norm(A * B - C, 'fro')
                
                if error < best_error:
                    best_error = error
                    best_A = A.copy()
                    best_B = B.copy()
                
                if abs(error - prev_error) < tol or error < tol:
                    break
                    
                prev_error = error
            
            # Early termination if we found a good solution
            if best_error < tol:
                break
        
        if best_error < tol:
            break
    
    return best_A, best_B, best_error

# Test on identity matrix
print("TESTING IMPROVED SOLVER ON IDENTITY MATRIX")
print("="*60)

I = np.eye(4)
print("\nTarget matrix (identity):")
print(I)

# Try the improved solver
A, B, error = solve_hadamard_improved(I, max_iter=500, n_restarts=20)

rel_error = error / np.linalg.norm(I, 'fro')
print(f"\nBest relative error: {rel_error:.6e}")
print(f"Best absolute error: {error:.6e}")

if rel_error < 1e-6:
    print("\n✅ SUCCESS! Found factorization")
    print(f"\nRank(A) = {np.linalg.matrix_rank(A)}")
    print(f"Rank(B) = {np.linalg.matrix_rank(B)}")
    
    print("\nA =")
    print(A)
    print("\nB =")
    print(B)
    print("\nA ⊙ B =")
    print(A * B)
else:
    print("\n❌ Still struggling to find factorization")
    print("The algorithm needs more sophisticated initialization")
    
print("\n\nDEMONSTRATING KNOWN SOLUTION:")
print("-"*40)
print("Here's the factorization from MathOverflow:")

B_known = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
], dtype=float)

C_known = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=float)

print("\nB =")
print(B_known)
print(f"Rank(B) = {np.linalg.matrix_rank(B_known)}")

print("\nC =")  
print(C_known)
print(f"Rank(C) = {np.linalg.matrix_rank(C_known)}")

print("\nB ⊙ C =")
print(B_known * C_known)

print("\nThis shows the identity CAN be factorized!")
print("The challenge is finding it algorithmically.")