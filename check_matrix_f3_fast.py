#!/usr/bin/env python3
"""
Fast F3 Hadamard factorization checker using Numba optimization.
Uses a two-stage approach: NumPy array for iteration, dictionary for lookups.
"""

import numpy as np
import sys
from typing import Tuple, Optional, Dict, Set
from itertools import product, combinations
import time
import pickle

try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available. Install with: pip install numba")
    print("Falling back to pure Python version.")

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

def generate_all_rank2_matrices():
    """Generate all rank-2 4×4 matrices over F3 using {-1, 0, 1}."""
    rank2_matrices_dict = {}
    rank2_matrices_list = []
    
    print("Generating all rank-2 matrices over F3...")
    count = 0
    
    for basis_rows in combinations(range(4), 2):
        i1, i2 = basis_rows
        
        for row1_vals in product([-1, 0, 1], repeat=4):
            row1 = np.array(row1_vals)
            if np.all(row1 == 0):
                continue
                
            for row2_vals in product([-1, 0, 1], repeat=4):
                row2 = np.array(row2_vals)
                
                if np.all(row2 == 0):
                    continue
                
                if np.any(row1 != 0):
                    nz_idx = np.nonzero(row1)[0][0]
                    if row1[nz_idx] != 0:
                        potential_mult = row2[nz_idx] / row1[nz_idx]
                        if np.allclose(row2, potential_mult * row1):
                            continue
                
                for coeffs in product([-1, 0, 1], repeat=4):
                    a, b, c, d = coeffs
                    
                    M = np.zeros((4, 4), dtype=np.int8)
                    M[i1] = row1
                    M[i2] = row2
                    
                    other_rows = [r for r in range(4) if r not in basis_rows]
                    M[other_rows[0]] = mod3(a * row1 + b * row2)
                    M[other_rows[1]] = mod3(c * row1 + d * row2)
                    
                    M = mod3(M)
                    if rank_f3(M) == 2:
                        key = matrix_to_tuple(M)
                        if key not in rank2_matrices_dict:
                            rank2_matrices_dict[key] = len(rank2_matrices_list)
                            rank2_matrices_list.append(M.copy())
                            count += 1
                            
                            if count % 10000 == 0:
                                print(f"  Generated {count} unique rank-2 matrices...")
    
    print(f"Total unique rank-2 matrices: {len(rank2_matrices_list)}")
    
    # Convert list to numpy array for fast iteration
    rank2_array = np.array(rank2_matrices_list, dtype=np.int8)
    
    return rank2_array, rank2_matrices_dict

if HAS_NUMBA:
    @jit(nopython=True)
    def hadamard_divide_f3_numba(M, A, result):
        """
        Compute Hadamard division M / A in F3 using {-1, 0, 1}.
        Returns False if division would involve dividing by zero.
        Modifies result array in-place.
        """
        for i in range(4):
            for j in range(4):
                if A[i, j] == 0:
                    if M[i, j] != 0:
                        return False  # Can't divide non-zero by zero
                    result[i, j] = 0
                else:
                    # In F3 with {-1, 0, 1}: x / y = x * y
                    result[i, j] = M[i, j] * A[i, j]
        return True
    
    @jit(nopython=True, parallel=True)
    def fast_check_factorization(M, rank2_array):
        """
        Fast factorization check using Numba.
        Returns index of matching B matrix, or -1 if not found.
        """
        n_matrices = rank2_array.shape[0]
        result = np.empty((4, 4), dtype=np.int8)
        
        for idx in prange(n_matrices):
            A = rank2_array[idx]
            
            # Try to compute B = M / A
            if hadamard_divide_f3_numba(M, A, result):
                # Check if result equals any rank-2 matrix
                for idx2 in range(n_matrices):
                    B = rank2_array[idx2]
                    
                    # Check if result equals B
                    match = True
                    for i in range(4):
                        for j in range(4):
                            if result[i, j] != B[i, j]:
                                match = False
                                break
                        if not match:
                            break
                    
                    if match:
                        # Found factorization: return indices
                        return idx * n_matrices + idx2
        
        return -1

def check_expressibility_f3_fast(M, rank2_array, rank2_dict):
    """Fast check if matrix M can be expressed as Hadamard product of rank-2 matrices."""
    M = mod3(M)
    
    print(f"Checking matrix over F3 (fast version with Numba):")
    print(M)
    
    m_rank = rank_f3(M)
    print(f"\nRank: {m_rank}")
    
    if m_rank <= 2:
        print("Matrix has rank ≤ 2, so it's trivially expressible.")
        return True, (M, np.ones((4, 4), dtype=int))
    
    print("\nSearching for factorization...")
    
    if HAS_NUMBA:
        # Use Numba-accelerated search
        M_int8 = M.astype(np.int8)
        result_idx = fast_check_factorization(M_int8, rank2_array)
        
        if result_idx >= 0:
            # Decode indices
            n_matrices = rank2_array.shape[0]
            idx_a = result_idx // n_matrices
            idx_b = result_idx % n_matrices
            
            A = rank2_array[idx_a]
            B = rank2_array[idx_b]
            
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
    else:
        # Fallback to pure Python
        checked = 0
        result = np.zeros((4, 4), dtype=np.int8)
        
        for idx, A in enumerate(rank2_array):
            checked += 1
            if checked % 50000 == 0:
                print(f"  Checked {checked}/{len(rank2_array)} matrices...")
            
            # Try to compute B = M / A
            can_divide = True
            for i in range(4):
                for j in range(4):
                    if A[i, j] == 0:
                        if M[i, j] != 0:
                            can_divide = False
                            break
                        result[i, j] = 0
                    else:
                        result[i, j] = M[i, j] * A[i, j]
                if not can_divide:
                    break
            
            if not can_divide:
                continue
            
            # Check if B is rank-2
            b_tuple = matrix_to_tuple(result)
            if b_tuple in rank2_dict:
                print(f"\nFound factorization!")
                print(f"A = \n{A}")
                print(f"B = \n{result}")
                
                # Verify
                product = A * result
                if np.array_equal(product, M):
                    print("Verification: A ∘ B = M ✓")
                    return True, (A, result)
                else:
                    print("ERROR: Verification failed!")
                    return False, None
    
    print("\nNo factorization found.")
    print("This matrix is a COUNTEREXAMPLE for F3")
    return False, None

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--example':
            print("Example usage:")
            print("python check_matrix_f3_fast.py")
            print("Then enter a 4x4 matrix over F3, e.g.:")
            print("1 2 0 1")
            print("2 1 1 0")
            print("0 1 2 2")
            print("1 0 2 1")
            return
        elif sys.argv[1] == '--generate-only':
            rank2_array, rank2_dict = generate_all_rank2_matrices()
            with open('rank2_matrices_f3_fast.pkl', 'wb') as f:
                pickle.dump((rank2_array, rank2_dict), f)
            print("Saved rank-2 matrices to rank2_matrices_f3_fast.pkl")
            return
        elif sys.argv[1] == '--benchmark' and HAS_NUMBA:
            print("Benchmarking Numba performance...")
            
            # Generate test data
            print("Generating rank-2 matrices...")
            rank2_array, rank2_dict = generate_all_rank2_matrices()
            
            # Create a test matrix
            test_matrix = mod3(np.random.randint(0, 3, (4, 4)))
            
            # Warm up Numba
            print("Warming up Numba...")
            _ = fast_check_factorization(test_matrix.astype(np.int8), rank2_array[:100])
            
            # Benchmark
            print("Running benchmark...")
            start = time.time()
            result = fast_check_factorization(test_matrix.astype(np.int8), rank2_array)
            numba_time = time.time() - start
            
            print(f"Numba search time: {numba_time:.3f}s")
            print(f"Matrices checked: {len(rank2_array)}")
            print(f"Rate: {len(rank2_array)/numba_time:.0f} matrices/second")
            return
    
    # Try to load precomputed rank-2 matrices
    try:
        print("Loading precomputed rank-2 matrices...")
        with open('rank2_matrices_f3_fast.pkl', 'rb') as f:
            rank2_array, rank2_dict = pickle.load(f)
        print(f"Loaded {len(rank2_array)} rank-2 matrices")
    except FileNotFoundError:
        print("No precomputed matrices found. Generating...")
        start_time = time.time()
        rank2_array, rank2_dict = generate_all_rank2_matrices()
        print(f"Generation took {time.time() - start_time:.2f} seconds")
        
        with open('rank2_matrices_f3_fast.pkl', 'wb') as f:
            pickle.dump((rank2_array, rank2_dict), f)
        print("Saved rank-2 matrices for future use")
    
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
    is_expressible, factorization = check_expressibility_f3_fast(M, rank2_array, rank2_dict)
    print(f"\nSearch took {time.time() - start_time:.3f} seconds")
    
    if is_expressible:
        print("\nResult: This matrix is EXPRESSIBLE as A ∘ B with rank(A) = rank(B) = 2")
    else:
        print("\nResult: This matrix is a COUNTEREXAMPLE")
        print("It cannot be expressed as A ∘ B with rank(A), rank(B) ≤ 2 over F3")

if __name__ == "__main__":
    main()