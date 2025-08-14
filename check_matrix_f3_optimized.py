#!/usr/bin/env python3
"""
Optimized F3 Hadamard factorization checker.
Simple optimizations that work with existing data format.
"""

import numpy as np
import sys
import time
import pickle
from typing import Dict

try:
    import numba
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not available. Install with: pip install numba")

def mod3(x):
    """Convert to {-1, 0, 1} representation."""
    x = x % 3
    return np.where(x == 2, -1, x)

def rank_f3(M):
    """Compute rank of matrix over F_3 using {-1, 0, 1} representation."""
    A = mod3(M.copy())
    n = A.shape[0]
    rank = 0
    
    for col in range(n):
        found = False
        for row in range(rank, n):
            if A[row, col] != 0:
                if row != rank:
                    A[[rank, row]] = A[[row, rank]]
                
                if A[rank, col] == -1:
                    A[rank] = -A[rank]
                
                for r in range(n):
                    if r != rank and A[r, col] != 0:
                        A[r] = mod3(A[r] - A[r, col] * A[rank])
                rank += 1
                found = True
                break
    return rank

def matrix_to_tuple(M):
    """Convert matrix to tuple for hashing."""
    return tuple(M.flatten())

if HAS_NUMBA:
    @njit
    def can_divide_and_get_quotient(M, A, B_result):
        """
        Check if M can be divided by A and compute quotient.
        Returns True if division is valid, False otherwise.
        Modifies B_result in-place with the quotient.
        """
        for i in range(4):
            for j in range(4):
                if A[i, j] == 0:
                    if M[i, j] != 0:
                        return False
                    B_result[i, j] = 0
                else:
                    # x / y = x * y in {-1,0,1} representation
                    B_result[i, j] = M[i, j] * A[i, j]
        return True
    
    @njit
    def matrices_equal(A, B):
        """Fast matrix equality check."""
        for i in range(4):
            for j in range(4):
                if A[i, j] != B[i, j]:
                    return False
        return True

def check_expressibility_f3_optimized(M, rank2_matrices: Dict[tuple, np.ndarray]):
    """Optimized check for F3 Hadamard factorization."""
    M = mod3(M).astype(np.int8)
    
    print(f"Checking matrix over F3 (optimized version):")
    print(M)
    
    m_rank = rank_f3(M)
    print(f"\nRank: {m_rank}")
    
    if m_rank <= 2:
        print("Matrix has rank ≤ 2, so it's trivially expressible.")
        return True, (M, np.ones((4, 4), dtype=int))
    
    print("\nSearching for factorization...")
    
    # Convert dictionary to list for iteration (keep dict for lookup)
    rank2_list = list(rank2_matrices.values())
    rank2_array = np.array(rank2_list, dtype=np.int8)
    
    # Pre-allocate result array
    B_temp = np.zeros((4, 4), dtype=np.int8)
    
    checked = 0
    batch_size = 10000
    
    if HAS_NUMBA:
        # Process in batches with Numba
        for start_idx in range(0, len(rank2_array), batch_size):
            end_idx = min(start_idx + batch_size, len(rank2_array))
            batch = rank2_array[start_idx:end_idx]
            
            for idx_in_batch, A in enumerate(batch):
                if can_divide_and_get_quotient(M, A, B_temp):
                    # Check if B_temp is rank-2
                    b_tuple = matrix_to_tuple(B_temp)
                    if b_tuple in rank2_matrices:
                        print(f"\nFound factorization!")
                        print(f"A = \n{A}")
                        print(f"B = \n{B_temp}")
                        
                        # Verify
                        product = A * B_temp
                        if np.array_equal(product, M):
                            print("Verification: A ∘ B = M ✓")
                            return True, (A, B_temp.copy())
                        else:
                            print("ERROR: Verification failed!")
                            return False, None
            
            checked += (end_idx - start_idx)
            if checked % 50000 == 0:
                print(f"  Checked {checked}/{len(rank2_array)} matrices...")
    else:
        # Fallback without Numba
        for A in rank2_list:
            checked += 1
            if checked % 50000 == 0:
                print(f"  Checked {checked}/{len(rank2_list)} matrices...")
            
            # Check compatibility
            can_divide = True
            for i in range(4):
                for j in range(4):
                    if A[i, j] == 0 and M[i, j] != 0:
                        can_divide = False
                        break
                if not can_divide:
                    break
            
            if not can_divide:
                continue
            
            # Compute quotient
            B = np.zeros((4, 4), dtype=np.int8)
            for i in range(4):
                for j in range(4):
                    if A[i, j] == 0:
                        B[i, j] = 0
                    else:
                        B[i, j] = M[i, j] * A[i, j]
            
            # Check if B is rank-2
            b_tuple = matrix_to_tuple(B)
            if b_tuple in rank2_matrices:
                print(f"\nFound factorization!")
                print(f"A = \n{A}")
                print(f"B = \n{B}")
                
                # Verify
                product = A * B
                if np.array_equal(product, M):
                    print("Verification: A ∘ B = M ✓")
                    return True, (A, B)
                else:
                    print("ERROR: Verification failed!")
                    return False, None
    
    print("\nNo factorization found.")
    print("This matrix is a COUNTEREXAMPLE for F3")
    return False, None

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        print("Benchmarking optimized version...")
        
        # Load precomputed matrices
        try:
            with open('rank2_matrices_f3.pkl', 'rb') as f:
                rank2_matrices = pickle.load(f)
            print(f"Loaded {len(rank2_matrices)} rank-2 matrices")
        except FileNotFoundError:
            print("Please run check_matrix_f3.py first to generate matrices")
            return
        
        # Test with identity matrix
        test_matrix = np.eye(4, dtype=int)
        
        # Warm up Numba if available
        if HAS_NUMBA:
            print("Warming up Numba...")
            M_test = mod3(test_matrix).astype(np.int8)
            A_test = list(rank2_matrices.values())[0].astype(np.int8)
            B_test = np.zeros((4, 4), dtype=np.int8)
            _ = can_divide_and_get_quotient(M_test, A_test, B_test)
        
        # Benchmark
        print("Running benchmark...")
        start = time.time()
        is_expressible, _ = check_expressibility_f3_optimized(test_matrix, rank2_matrices)
        search_time = time.time() - start
        
        print(f"\nResults:")
        print(f"Search time: {search_time:.3f}s")
        print(f"Found factorization: {is_expressible}")
        return
    
    # Load precomputed matrices
    try:
        print("Loading precomputed rank-2 matrices...")
        with open('rank2_matrices_f3.pkl', 'rb') as f:
            rank2_matrices = pickle.load(f)
        print(f"Loaded {len(rank2_matrices)} rank-2 matrices")
    except FileNotFoundError:
        print("Error: rank2_matrices_f3.pkl not found.")
        print("Please run: python check_matrix_f3.py --generate-only")
        return
    
    print("\nEnter a 4×4 matrix over F3 (4 lines, 4 values each):")
    print("Values should be 0, 1, or 2")
    
    M = np.zeros((4, 4), dtype=int)
    try:
        for i in range(4):
            line = input().strip().split()
            if len(line) != 4:
                print(f"Error: Row {i+1} must have exactly 4 values")
                return
            for j in range(4):
                val = int(line[j])
                if val not in [0, 1, 2]:
                    print(f"Error: Values must be 0, 1, or 2, got {val}")
                    return
                M[i, j] = val
    except (ValueError, EOFError) as e:
        print(f"Error reading matrix: {e}")
        return
    
    print()
    start_time = time.time()
    is_expressible, factorization = check_expressibility_f3_optimized(M, rank2_matrices)
    print(f"\nSearch took {time.time() - start_time:.3f} seconds")
    
    if is_expressible:
        print("\nResult: This matrix is EXPRESSIBLE as A ∘ B with rank(A) = rank(B) = 2")
    else:
        print("\nResult: This matrix is a COUNTEREXAMPLE")
        print("It cannot be expressed as A ∘ B with rank(A), rank(B) ≤ 2 over F3")

if __name__ == "__main__":
    main()