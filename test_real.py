#!/usr/bin/env python3
"""
Test factorization over the real numbers using numerical optimization.

Note: The real case is more subtle due to the interaction between
zero constraints and rank constraints.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def create_rank2_matrix(params):
    """Create a rank-2 matrix from parameters."""
    A = np.zeros((4, 4))
    # First two rows are independent
    A[0, :] = params[0:4]
    A[1, :] = params[4:8]
    # Last two rows are linear combinations
    A[2, :] = params[8] * A[0, :] + params[9] * A[1, :]
    A[3, :] = params[10] * A[0, :] + params[11] * A[1, :]
    return A

def test_real_factorization(M, max_trials=100, verbose=True):
    """
    Test if matrix M can be factored as A ∘ B with rank(A), rank(B) ≤ 2.
    """
    if verbose:
        print(f"\nTesting matrix:")
        print(M.astype(int))
        print(f"Rank: {np.linalg.matrix_rank(M)}")
        print(f"Number of zeros: {np.sum(M == 0)}")
    
    zero_mask = (M != 0).astype(float)
    
    def objective(x):
        A = create_rank2_matrix(x[:12])
        B = create_rank2_matrix(x[12:24])
        
        # Apply zero constraints
        A_masked = A * zero_mask
        B_masked = B * zero_mask
        
        # Compute Hadamard product
        C = A_masked * B_masked
        
        # Reconstruction error
        error = np.sum((C - M)**2)
        
        # Check actual ranks after masking
        rank_A = np.linalg.matrix_rank(A_masked, tol=1e-10)
        rank_B = np.linalg.matrix_rank(B_masked, tol=1e-10)
        
        # Add penalty for high rank
        if rank_A > 2:
            error += 1e6 * (rank_A - 2)**2
        if rank_B > 2:
            error += 1e6 * (rank_B - 2)**2
        
        return error
    
    best_error = float('inf')
    best_result = None
    
    # Try multiple random initializations
    for trial in range(max_trials):
        if trial % 20 == 0 and verbose:
            print(f"Trial {trial}/{max_trials}...")
        
        # Random initialization
        x0 = np.random.randn(24) * 2
        
        # Local optimization
        result = minimize(objective, x0, method='Powell', 
                         options={'maxiter': 5000, 'ftol': 1e-12})
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result
            
            # Check if we found a good solution
            if best_error < 1e-8:
                A = create_rank2_matrix(best_result.x[:12])
                B = create_rank2_matrix(best_result.x[12:24])
                A_masked = A * zero_mask
                B_masked = B * zero_mask
                
                rank_A = np.linalg.matrix_rank(A_masked, tol=1e-10)
                rank_B = np.linalg.matrix_rank(B_masked, tol=1e-10)
                
                if rank_A <= 2 and rank_B <= 2:
                    if verbose:
                        print(f"✓ Found factorization!")
                        print(f"  Ranks after masking: {rank_A}, {rank_B}")
                    return True, best_error
    
    if verbose:
        print(f"✗ No factorization found")
        print(f"  Best error: {best_error:.2e}")
        
        if best_result is not None:
            A = create_rank2_matrix(best_result.x[:12])
            B = create_rank2_matrix(best_result.x[12:24])
            A_masked = A * zero_mask
            B_masked = B * zero_mask
            
            rank_A = np.linalg.matrix_rank(A_masked, tol=1e-10)
            rank_B = np.linalg.matrix_rank(B_masked, tol=1e-10)
            print(f"  Best ranks achieved: {rank_A}, {rank_B}")
    
    return False, best_error

def main():
    print("TESTING FACTORIZATION OVER REAL NUMBERS")
    print("="*50)
    
    # Test known factorizable matrices
    print("\n1. Testing known factorizable matrices:")
    
    # Identity matrix
    I = np.eye(4)
    print("\nIdentity matrix:")
    can_factor, error = test_real_factorization(I, max_trials=50)
    
    # Our counterexample
    print("\n2. Testing our main counterexample:")
    counterexample = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=float)
    
    can_factor_ce, error_ce = test_real_factorization(counterexample, max_trials=50)
    
    # Additional counterexamples
    print("\n3. Testing additional counterexamples:")
    
    ce2 = np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=float)
    
    can_factor_ce2, error_ce2 = test_real_factorization(ce2, max_trials=50)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Identity matrix: {'CAN' if can_factor else 'CANNOT'} be factored")
    print(f"Counterexample 1: {'CAN' if can_factor_ce else 'CANNOT'} be factored")
    print(f"Counterexample 2: {'CAN' if can_factor_ce2 else 'CANNOT'} be factored")
    
    print("\nIMPORTANT NOTE:")
    print("The numerical optimization may find low reconstruction error,")
    print("but the key is whether the factors have rank ≤ 2 AFTER applying")
    print("the zero constraints. Our counterexamples require rank > 2.")

if __name__ == "__main__":
    main()