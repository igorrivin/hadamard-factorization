#!/usr/bin/env python3
"""
Solve Hadamard factorization over the reals using optimization.
Given a 4x4 matrix C, find rank-2 matrices A, B such that C = A ∘ B.
"""

import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def parameterize_rank2_matrix(r1: np.ndarray, r2: np.ndarray, 
                              c3: float, d3: float, c4: float, d4: float) -> np.ndarray:
    """
    Parameterize a rank-2 4x4 matrix with two base rows and linear combinations.
    
    Matrix has rows:
    - r1
    - r2  
    - c3*r1 + d3*r2
    - c4*r1 + d4*r2
    """
    A = np.zeros((4, 4))
    A[0] = r1
    A[1] = r2
    A[2] = c3 * r1 + d3 * r2
    A[3] = c4 * r1 + d4 * r2
    return A

def params_to_matrices(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert parameter vector to two rank-2 matrices.
    
    params = [r1_A (4), r2_A (4), c3_A, d3_A, c4_A, d4_A,
              r1_B (4), r2_B (4), c3_B, d3_B, c4_B, d4_B]
    Total: 24 parameters
    """
    # First matrix A
    r1_A = params[0:4]
    r2_A = params[4:8]
    c3_A, d3_A, c4_A, d4_A = params[8:12]
    
    # Second matrix B
    r1_B = params[12:16]
    r2_B = params[16:20]
    c3_B, d3_B, c4_B, d4_B = params[20:24]
    
    A = parameterize_rank2_matrix(r1_A, r2_A, c3_A, d3_A, c4_A, d4_A)
    B = parameterize_rank2_matrix(r1_B, r2_B, c3_B, d3_B, c4_B, d4_B)
    
    return A, B

def objective_function(params: np.ndarray, C: np.ndarray) -> float:
    """Objective: ||A ∘ B - C||_F^2"""
    A, B = params_to_matrices(params)
    diff = A * B - C  # Hadamard product
    return np.sum(diff**2)

def gradient(params: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute gradient of objective function."""
    eps = 1e-8
    grad = np.zeros_like(params)
    f0 = objective_function(params, C)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = objective_function(params_plus, C)
        grad[i] = (f_plus - f0) / eps
    
    return grad

def residual_vector(params: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residual vector for least squares: vec(A ∘ B - C)"""
    A, B = params_to_matrices(params)
    diff = A * B - C
    return diff.flatten()

def solve_hadamard_factorization(C: np.ndarray, 
                                 method: str = 'least_squares',
                                 n_trials: int = 10,
                                 verbose: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Find rank-2 matrices A, B such that C ≈ A ∘ B.
    
    Args:
        C: Target 4x4 matrix
        method: 'least_squares', 'newton', or 'gradient'
        n_trials: Number of random initializations
        verbose: Print progress
        
    Returns:
        A, B, error (or None if failed)
    """
    best_error = float('inf')
    best_A = None
    best_B = None
    
    for trial in range(n_trials):
        # Random initialization
        if trial == 0:
            # First trial: smart initialization based on SVD
            U, S, Vt = np.linalg.svd(C)
            scale = np.sqrt(np.abs(S[0]))
            params = np.random.randn(24) * scale
        else:
            # Random initialization
            params = np.random.randn(24) * np.std(C)
        
        try:
            if method == 'least_squares':
                # Levenberg-Marquardt
                result = least_squares(
                    lambda p: residual_vector(p, C),
                    params,
                    method='lm',
                    max_nfev=10000
                )
                final_params = result.x
                
            elif method == 'gradient':
                # L-BFGS-B
                result = minimize(
                    lambda p: objective_function(p, C),
                    params,
                    method='L-BFGS-B',
                    jac=lambda p: gradient(p, C),
                    options={'maxiter': 1000}
                )
                final_params = result.x
                
            else:  # newton
                # Trust region Newton
                result = minimize(
                    lambda p: objective_function(p, C),
                    params,
                    method='trust-ncg',
                    jac=lambda p: gradient(p, C),
                    options={'maxiter': 1000}
                )
                final_params = result.x
            
            # Extract solution
            A, B = params_to_matrices(final_params)
            error = np.linalg.norm(A * B - C, 'fro')
            
            if error < best_error:
                best_error = error
                best_A = A
                best_B = B
                
            if verbose and trial % max(1, n_trials // 10) == 0:
                print(f"Trial {trial}: error = {error:.6f}, best = {best_error:.6f}")
                
        except Exception as e:
            if verbose:
                print(f"Trial {trial} failed: {e}")
    
    if best_A is None:
        return None
    
    # Verify rank
    rank_A = np.linalg.matrix_rank(best_A, tol=1e-10)
    rank_B = np.linalg.matrix_rank(best_B, tol=1e-10)
    
    if verbose:
        print(f"\nBest solution:")
        print(f"  Error: {best_error:.6e}")
        print(f"  Rank(A): {rank_A}, Rank(B): {rank_B}")
        print(f"  Relative error: {best_error / np.linalg.norm(C, 'fro'):.6e}")
    
    return best_A, best_B, best_error

def test_on_known_factorizable():
    """Test on a matrix we know is factorizable."""
    print("="*60)
    print("TEST 1: Known factorizable matrix")
    print("="*60)
    
    # Create rank-2 matrices
    A_true = np.array([[1, 2, 3, 4],
                       [2, 1, 4, 3],
                       [3, 3, 7, 7],
                       [4, 4, 10, 10]])
    
    B_true = np.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 1, 1, 1],
                       [2, 2, 2, 2]])
    
    C = A_true * B_true
    
    print("Target matrix C:")
    print(C)
    print(f"\nRank(A_true) = {np.linalg.matrix_rank(A_true)}")
    print(f"Rank(B_true) = {np.linalg.matrix_rank(B_true)}")
    
    # Try to recover
    result = solve_hadamard_factorization(C, method='least_squares', n_trials=5)
    
    if result:
        A, B, error = result
        print("\nRecovered vs True:")
        print(f"||A*B - C|| = {error:.6e}")
        print(f"||A - A_true|| = {np.linalg.norm(A - A_true):.6f}")
        print(f"||B - B_true|| = {np.linalg.norm(B - B_true):.6f}")

def test_on_random_matrices():
    """Test on random matrices."""
    print("\n" + "="*60)
    print("TEST 2: Random matrices")
    print("="*60)
    
    np.random.seed(42)
    n_tests = 10
    success_count = 0
    
    for i in range(n_tests):
        # Generate random matrix
        C = np.random.randn(4, 4)
        
        print(f"\nTest {i+1}:")
        result = solve_hadamard_factorization(C, method='least_squares', 
                                              n_trials=5, verbose=False)
        
        if result:
            A, B, error = result
            rel_error = error / np.linalg.norm(C, 'fro')
            print(f"  Relative error: {rel_error:.6e}")
            print(f"  Ranks: {np.linalg.matrix_rank(A)}, {np.linalg.matrix_rank(B)}")
            
            if rel_error < 1e-6:
                success_count += 1
                print("  ✓ Success!")
            else:
                print("  ✗ Large error")
    
    print(f"\nSuccess rate: {success_count}/{n_tests} = {success_count/n_tests*100:.1f}%")

def test_zero_free_matrices():
    """Test specifically on zero-free matrices."""
    print("\n" + "="*60)
    print("TEST 3: Zero-free matrices (dense)")
    print("="*60)
    
    np.random.seed(42)
    n_tests = 10
    success_count = 0
    
    for i in range(n_tests):
        # Generate zero-free matrix
        C = np.random.randn(4, 4)
        C = np.sign(C) * (np.abs(C) + 1)  # Ensure no zeros
        
        print(f"\nTest {i+1} (zeros: {np.sum(C == 0)}):")
        result = solve_hadamard_factorization(C, method='least_squares',
                                              n_trials=10, verbose=False)
        
        if result:
            A, B, error = result
            rel_error = error / np.linalg.norm(C, 'fro')
            print(f"  Relative error: {rel_error:.6e}")
            
            if rel_error < 1e-6:
                success_count += 1
                print("  ✓ Success!")
                
                # Check if A, B are also zero-free
                zeros_A = np.sum(np.abs(A) < 1e-10)
                zeros_B = np.sum(np.abs(B) < 1e-10)
                print(f"  Zeros in A: {zeros_A}, Zeros in B: {zeros_B}")
            else:
                print("  ✗ Large error")
    
    print(f"\nSuccess rate for zero-free: {success_count}/{n_tests} = {success_count/n_tests*100:.1f}%")

def analyze_landscape(C: np.ndarray):
    """Analyze the optimization landscape for a given target matrix."""
    print("\n" + "="*60)
    print("OPTIMIZATION LANDSCAPE ANALYSIS")
    print("="*60)
    
    # Try multiple methods
    methods = ['least_squares', 'gradient', 'newton']
    results = {}
    
    for method in methods:
        print(f"\nMethod: {method}")
        result = solve_hadamard_factorization(C, method=method, 
                                              n_trials=20, verbose=False)
        if result:
            A, B, error = result
            results[method] = error
            print(f"  Final error: {error:.6e}")
            print(f"  Ranks: {np.linalg.matrix_rank(A)}, {np.linalg.matrix_rank(B)}")
    
    # Plot convergence curves (would need to modify solve function to return history)
    return results

if __name__ == "__main__":
    print("HADAMARD FACTORIZATION OVER THE REALS")
    print("="*60)
    
    # Run tests
    test_on_known_factorizable()
    test_on_random_matrices()
    test_zero_free_matrices()
    
    # Analyze a specific challenging case
    print("\n" + "="*60)
    print("CHALLENGING CASE: Symmetric zero-free matrix")
    print("="*60)
    
    C_challenge = np.array([[2, 3, 4, 5],
                            [3, 2, 5, 4],
                            [4, 5, 2, 3],
                            [5, 4, 3, 2]])
    
    print("Target matrix:")
    print(C_challenge)
    print(f"Rank: {np.linalg.matrix_rank(C_challenge)}")
    print(f"Zeros: {np.sum(C_challenge == 0)}")
    
    analyze_landscape(C_challenge)