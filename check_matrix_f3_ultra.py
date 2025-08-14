#!/usr/bin/env python3
"""
Ultra-fast F3 Hadamard factorization checker using optimized Numba code.
Key optimizations:
1. Matrix-to-integer encoding for fast comparison
2. Sorted array with binary search
3. Optimized Numba kernels without failed parallelization
4. Bit-packing for the {-1,0,1} values
"""

import numpy as np
import sys
import time
import pickle
from typing import Tuple, Optional, Dict
from itertools import product, combinations

try:
    import numba
    from numba import jit, njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available. Install with: pip install numba")
    sys.exit(1)

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

@njit
def matrix_to_int64(M):
    """
    Encode a 4x4 matrix with values in {-1,0,1} as a 64-bit integer.
    Each element uses 2 bits: 00 for 0, 01 for 1, 11 for -1
    """
    result = np.int64(0)
    shift = 0
    for i in range(4):
        for j in range(4):
            val = M[i, j]
            if val == 1:
                bits = 1
            elif val == -1:
                bits = 3
            else:  # val == 0
                bits = 0
            result |= np.int64(bits) << shift
            shift += 2
    return result

@njit
def int64_to_matrix(val, result):
    """Decode a 64-bit integer back to a 4x4 matrix. Modifies result in-place."""
    shift = 0
    for i in range(4):
        for j in range(4):
            bits = (val >> shift) & 3
            if bits == 1:
                result[i, j] = 1
            elif bits == 3:
                result[i, j] = -1
            else:
                result[i, j] = 0
            shift += 2

@njit
def binary_search(arr, target):
    """Binary search in sorted array. Returns True if found."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

@njit
def hadamard_divide_check_f3(M, A, B_temp, sorted_rank2_ints):
    """
    Try to divide M by A and check if result is rank-2.
    Returns (success, A_index, B_encoded) or (False, -1, -1)
    """
    # Check compatibility and compute B = M / A
    for i in range(4):
        for j in range(4):
            if A[i, j] == 0:
                if M[i, j] != 0:
                    return False, -1, -1
                B_temp[i, j] = 0
            else:
                # x / y = x * y in {-1,0,1} representation
                B_temp[i, j] = M[i, j] * A[i, j]
    
    # Encode B and check if it's rank-2
    B_encoded = matrix_to_int64(B_temp)
    if binary_search(sorted_rank2_ints, B_encoded):
        return True, 0, B_encoded
    
    return False, -1, -1

@njit
def fast_factorization_search(M, rank2_array, sorted_rank2_ints):
    """
    Ultra-fast search for factorization.
    Returns (found, A_idx, B_encoded) or (False, -1, -1)
    """
    n_matrices = len(rank2_array)
    B_temp = np.empty((4, 4), dtype=np.int8)
    
    for idx in range(n_matrices):
        A = rank2_array[idx]
        
        found, _, B_encoded = hadamard_divide_check_f3(M, A, B_temp, sorted_rank2_ints)
        if found:
            return True, idx, B_encoded
    
    return False, -1, -1

def generate_all_rank2_matrices_ultra():
    """Generate rank-2 matrices and return both array and sorted integer encoding."""
    rank2_matrices_set = set()
    rank2_matrices_list = []
    
    print("Generating all rank-2 matrices over F3...")
    count = 0
    
    for basis_rows in combinations(range(4), 2):
        i1, i2 = basis_rows
        
        for row1_vals in product([-1, 0, 1], repeat=4):
            row1 = np.array(row1_vals, dtype=np.int8)
            if np.all(row1 == 0):
                continue
                
            for row2_vals in product([-1, 0, 1], repeat=4):
                row2 = np.array(row2_vals, dtype=np.int8)
                
                if np.all(row2 == 0):
                    continue
                
                # Quick linear independence check
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
                    M[other_rows[0]] = mod3(a * row1 + b * row2).astype(np.int8)
                    M[other_rows[1]] = mod3(c * row1 + d * row2).astype(np.int8)
                    
                    if rank_f3(M) == 2:
                        # Use integer encoding for uniqueness check
                        encoded = matrix_to_int64(M)
                        if encoded not in rank2_matrices_set:
                            rank2_matrices_set.add(encoded)
                            rank2_matrices_list.append(M.copy())
                            count += 1
                            
                            if count % 10000 == 0:
                                print(f"  Generated {count} unique rank-2 matrices...")
    
    print(f"Total unique rank-2 matrices: {len(rank2_matrices_list)}")
    
    # Convert to arrays
    rank2_array = np.array(rank2_matrices_list, dtype=np.int8)
    
    # Create sorted integer array for binary search
    rank2_ints = np.array([matrix_to_int64(m) for m in rank2_array], dtype=np.int64)
    sorted_indices = np.argsort(rank2_ints)
    sorted_rank2_ints = rank2_ints[sorted_indices]
    
    # Also create reverse mapping from int to original index
    int_to_idx = {int(rank2_ints[i]): i for i in range(len(rank2_ints))}
    
    return rank2_array, sorted_rank2_ints, int_to_idx

def check_expressibility_f3_ultra(M, rank2_array, sorted_rank2_ints, int_to_idx):
    """Ultra-fast check for F3 Hadamard factorization."""
    M = mod3(M).astype(np.int8)
    
    print(f"Checking matrix over F3 (ultra-fast version):")
    print(M)
    
    m_rank = rank_f3(M)
    print(f"\nRank: {m_rank}")
    
    if m_rank <= 2:
        print("Matrix has rank ≤ 2, so it's trivially expressible.")
        return True, (M, np.ones((4, 4), dtype=int))
    
    print("\nSearching for factorization...")
    
    # Run the optimized search
    found, a_idx, b_encoded = fast_factorization_search(M, rank2_array, sorted_rank2_ints)
    
    if found:
        A = rank2_array[a_idx]
        
        # Reconstruct B from encoding
        B = np.empty((4, 4), dtype=np.int8)
        int64_to_matrix(b_encoded, B)
        
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
    if len(sys.argv) > 1:
        if sys.argv[1] == '--example':
            print("Example usage:")
            print("python check_matrix_f3_ultra.py")
            print("Then enter a 4x4 matrix over F3, e.g.:")
            print("1 2 0 1")
            print("2 1 1 0")
            print("0 1 2 2")
            print("1 0 2 1")
            return
        elif sys.argv[1] == '--generate-only':
            rank2_array, sorted_rank2_ints, int_to_idx = generate_all_rank2_matrices_ultra()
            with open('rank2_matrices_f3_ultra.pkl', 'wb') as f:
                pickle.dump((rank2_array, sorted_rank2_ints, int_to_idx), f)
            print("Saved rank-2 matrices to rank2_matrices_f3_ultra.pkl")
            return
        elif sys.argv[1] == '--benchmark':
            print("Benchmarking ultra-fast version...")
            
            # Generate or load data
            try:
                with open('rank2_matrices_f3_ultra.pkl', 'rb') as f:
                    rank2_array, sorted_rank2_ints, int_to_idx = pickle.load(f)
                print(f"Loaded {len(rank2_array)} rank-2 matrices")
            except FileNotFoundError:
                print("Generating rank-2 matrices...")
                rank2_array, sorted_rank2_ints, int_to_idx = generate_all_rank2_matrices_ultra()
            
            # Test with identity matrix (known counterexample)
            test_matrix = np.eye(4, dtype=np.int8)
            
            # Warm up Numba
            print("Warming up Numba...")
            _ = fast_factorization_search(test_matrix, rank2_array[:100], sorted_rank2_ints[:100])
            
            # Benchmark
            print("Running benchmark...")
            start = time.time()
            found, _, _ = fast_factorization_search(test_matrix, rank2_array, sorted_rank2_ints)
            search_time = time.time() - start
            
            print(f"\nResults:")
            print(f"Search time: {search_time:.3f}s")
            print(f"Matrices checked: {len(rank2_array)}")
            print(f"Rate: {len(rank2_array)/search_time:.0f} matrices/second")
            print(f"Found factorization: {found}")
            return
    
    # Try to load precomputed rank-2 matrices
    try:
        print("Loading precomputed rank-2 matrices...")
        with open('rank2_matrices_f3_ultra.pkl', 'rb') as f:
            rank2_array, sorted_rank2_ints, int_to_idx = pickle.load(f)
        print(f"Loaded {len(rank2_array)} rank-2 matrices")
    except FileNotFoundError:
        print("No precomputed matrices found. Generating...")
        start_time = time.time()
        rank2_array, sorted_rank2_ints, int_to_idx = generate_all_rank2_matrices_ultra()
        print(f"Generation took {time.time() - start_time:.2f} seconds")
        
        with open('rank2_matrices_f3_ultra.pkl', 'wb') as f:
            pickle.dump((rank2_array, sorted_rank2_ints, int_to_idx), f)
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
    is_expressible, factorization = check_expressibility_f3_ultra(M, rank2_array, sorted_rank2_ints, int_to_idx)
    print(f"\nSearch took {time.time() - start_time:.3f} seconds")
    
    if is_expressible:
        print("\nResult: This matrix is EXPRESSIBLE as A ∘ B with rank(A) = rank(B) = 2")
    else:
        print("\nResult: This matrix is a COUNTEREXAMPLE")
        print("It cannot be expressed as A ∘ B with rank(A), rank(B) ≤ 2 over F3")

if __name__ == "__main__":
    main()