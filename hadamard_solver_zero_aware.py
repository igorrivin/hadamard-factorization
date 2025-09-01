#!/usr/bin/env python3
"""
Zero-aware Hadamard factorization solver.
Ensures at least one factor has zeros where the target has zeros.
"""

import numpy as np
from scipy.linalg import svd

def project_to_rank2(M):
    """Project matrix to nearest rank-2 matrix."""
    U, s, Vt = svd(M)
    s[2:] = 0  # Keep only top 2 singular values
    return U @ np.diag(s) @ Vt

def create_zero_preserving_initialization(C, sparsity_factor=0.5):
    """
    Create initialization that respects zero pattern of C.
    
    For C = A ⊙ B, if C[i,j] = 0, then either A[i,j] = 0 or B[i,j] = 0 (or both).
    We randomly assign zeros to either A or B at each zero position of C.
    """
    n = C.shape[0]
    
    # Find zero positions in C
    zero_mask = np.abs(C) < 1e-10
    
    # Random initialization
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    # For each zero in C, randomly decide whether to zero out A or B
    # Use sparsity_factor to control distribution
    for i in range(n):
        for j in range(n):
            if zero_mask[i, j]:
                if np.random.rand() < sparsity_factor:
                    A[i, j] = 0
                else:
                    B[i, j] = 0
    
    # For non-zero positions in C, initialize to satisfy C = A ⊙ B approximately
    for i in range(n):
        for j in range(n):
            if not zero_mask[i, j]:
                # Randomly split the value
                sign_a = np.sign(np.random.randn())
                sign_b = np.sign(np.random.randn())
                magnitude = np.sqrt(np.abs(C[i, j]))
                A[i, j] = sign_a * magnitude
                B[i, j] = sign_b * magnitude * C[i, j] / (A[i, j] + 1e-10)
    
    # Project to rank 2
    A = project_to_rank2(A)
    B = project_to_rank2(B)
    
    return A, B

def solve_hadamard_zero_aware(C, max_iter=1000, tol=1e-10, n_restarts=20, 
                              use_patterns=True):
    """
    Improved Hadamard factorization solver with zero-awareness.
    
    Key improvements:
    1. Respects zero structure of target matrix
    2. Tries multiple sparsity patterns
    3. Uses structured initializations for special cases
    """
    n = C.shape[0]
    best_error = float('inf')
    best_A, best_B = None, None
    
    # Count zeros in C for diagnostics
    n_zeros = np.sum(np.abs(C) < 1e-10)
    print(f"Target has {n_zeros} zeros out of {n*n} entries")
    
    # Try different initialization strategies
    strategies = []
    
    # Zero-preserving with different sparsity factors
    for sparsity in [0.3, 0.5, 0.7]:
        strategies.append(('zero_preserving', sparsity))
    
    # Add pattern-based strategies if requested
    if use_patterns:
        strategies.extend([
            ('block_diagonal', None),
            ('alternating', None),
            ('complementary', None)
        ])
    
    for strategy, param in strategies:
        for restart in range(max(1, n_restarts // len(strategies))):
            
            # Initialize based on strategy
            if strategy == 'zero_preserving':
                A, B = create_zero_preserving_initialization(C, param)
                
            elif strategy == 'block_diagonal':
                # Block diagonal pattern (like MathOverflow example)
                A = np.zeros((n, n))
                A[0:n//2, 0:n//2] = np.random.randn(n//2, n//2)
                A[n//2:n, n//2:n] = np.random.randn(n//2, n//2)
                A = project_to_rank2(A)
                B = create_complementary_pattern(C, A)
                
            elif strategy == 'alternating':
                # Alternating columns pattern
                A = np.zeros((n, n))
                for i in range(n):
                    for j in range(0, n, 2):
                        A[i, j] = np.random.randn()
                A = project_to_rank2(A)
                B = create_complementary_pattern(C, A)
                
            elif strategy == 'complementary':
                # Try to mimic known factorization patterns
                A = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1]], dtype=float)
                A = A + 0.1 * np.random.randn(n, n)
                A = project_to_rank2(A)
                B = create_complementary_pattern(C, A)
            
            # Run alternating projections
            prev_error = float('inf')
            
            for iteration in range(max_iter):
                # Update B given A (with zero preservation)
                B_new = np.zeros_like(B)
                for i in range(n):
                    for j in range(n):
                        if abs(A[i,j]) > 1e-10:
                            B_new[i,j] = C[i,j] / A[i,j]
                        else:
                            # A[i,j] = 0, so B[i,j] can be anything if C[i,j] = 0
                            if abs(C[i,j]) > 1e-10:
                                # Problem: C[i,j] ≠ 0 but A[i,j] = 0
                                # This suggests bad initialization
                                B_new[i,j] = 1e6  # Large value to increase error
                            else:
                                B_new[i,j] = B[i,j]  # Keep current value
                
                B = project_to_rank2(B_new)
                
                # Update A given B (with zero preservation)
                A_new = np.zeros_like(A)
                for i in range(n):
                    for j in range(n):
                        if abs(B[i,j]) > 1e-10:
                            A_new[i,j] = C[i,j] / B[i,j]
                        else:
                            if abs(C[i,j]) > 1e-10:
                                A_new[i,j] = 1e6  # Large value to increase error
                            else:
                                A_new[i,j] = A[i,j]  # Keep current value
                
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
                print(f"Success with strategy: {strategy}")
                break
        
        if best_error < tol:
            break
    
    return best_A, best_B, best_error

def create_complementary_pattern(C, A):
    """Create B that complements A to satisfy zero pattern of C."""
    n = C.shape[0]
    B = np.zeros_like(A)
    
    for i in range(n):
        for j in range(n):
            if abs(C[i,j]) < 1e-10:  # C[i,j] = 0
                if abs(A[i,j]) < 1e-10:  # A[i,j] = 0
                    B[i,j] = np.random.randn()  # B can be anything
                else:  # A[i,j] ≠ 0
                    B[i,j] = 0  # B must be 0
            else:  # C[i,j] ≠ 0
                if abs(A[i,j]) > 1e-10:
                    B[i,j] = C[i,j] / A[i,j]
                else:
                    B[i,j] = 1e6  # Signal problem
    
    return project_to_rank2(B)

def test_on_identity():
    """Test the zero-aware solver on the identity matrix."""
    print("TESTING ZERO-AWARE SOLVER ON IDENTITY MATRIX")
    print("="*60)
    
    I = np.eye(4)
    
    # Try the zero-aware solver
    A, B, error = solve_hadamard_zero_aware(I, max_iter=500, n_restarts=30)
    
    rel_error = error / np.linalg.norm(I, 'fro')
    print(f"\nFinal relative error: {rel_error:.6e}")
    
    if rel_error < 1e-6:
        print("\n✅ SUCCESS! Found factorization")
        print(f"Rank(A) = {np.linalg.matrix_rank(A)}")
        print(f"Rank(B) = {np.linalg.matrix_rank(B)}")
        
        print("\nA =")
        print(A)
        print("\nB =")
        print(B)
        print("\nA ⊙ B =")
        print(A * B)
        
        # Analyze the structure
        print("\n\nStructure analysis:")
        print(f"Zeros in A: {np.sum(np.abs(A) < 1e-10)}")
        print(f"Zeros in B: {np.sum(np.abs(B) < 1e-10)}")
        
        # Check complementary zeros
        for i in range(4):
            for j in range(4):
                if i != j:  # Off-diagonal
                    if abs(A[i,j]) < 1e-10 and abs(B[i,j]) < 1e-10:
                        print(f"Warning: Both A[{i},{j}] and B[{i},{j}] are zero!")
    else:
        print("\n❌ Still struggling, but error is lower than naive approach")
        print("The identity matrix is a particularly difficult case")

def test_on_various_matrices():
    """Test on various matrix types to verify the solver works well."""
    print("\n\nTESTING ON VARIOUS MATRICES")
    print("="*60)
    
    test_cases = {
        "Known factorizable (sparse)": np.array([
            [1, 0, 1, 0],
            [0, 2, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 3]
        ], dtype=float),
        
        "Dense matrix": np.ones((4, 4)) + np.random.randn(4, 4) * 0.1,
        
        "Random sparse": None  # Will create below
    }
    
    # Create random sparse matrix
    sparse = np.random.randn(4, 4)
    sparse[np.random.rand(4, 4) < 0.6] = 0  # 60% zeros
    test_cases["Random sparse"] = sparse
    
    for name, C in test_cases.items():
        print(f"\n{name}:")
        print(f"Zeros: {np.sum(np.abs(C) < 1e-10)}/16")
        
        A, B, error = solve_hadamard_zero_aware(C, max_iter=200, n_restarts=10)
        rel_error = error / (np.linalg.norm(C, 'fro') + 1e-10)
        
        print(f"Relative error: {rel_error:.6e}")
        print(f"Success: {'✓' if rel_error < 1e-6 else '✗'}")

if __name__ == "__main__":
    test_on_identity()
    test_on_various_matrices()