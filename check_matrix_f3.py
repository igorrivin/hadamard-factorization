#!/usr/bin/env python3
"""
Check if a specific 4×4 F3 matrix can be expressed as a Hadamard product of two rank-2 matrices.
Uses {-1, 0, 1} representation for F3 for efficiency.
"""

import numpy as np
import sys
from typing import Tuple, Optional, Dict, Set
from itertools import product, combinations
import time

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
        # Find non-zero element in column
        found = False
        for row in range(rank, n):
            if A[row, col] != 0:
                if row != rank:
                    A[[rank, row]] = A[[row, rank]]
                
                # Make leading coefficient 1
                if A[rank, col] == -1:
                    A[rank] = -A[rank]  # Multiply by -1
                
                # Eliminate column
                for r in range(n):
                    if r != rank and A[r, col] != 0:
                        A[r] = mod3(A[r] - A[r, col] * A[rank])
                rank += 1
                found = True
                break
        if not found:
            continue
    return rank

def matrix_to_tuple(M):
    """Convert matrix to tuple for hashing."""
    return tuple(M.flatten())

def tuple_to_matrix(t):
    """Convert tuple back to 4x4 matrix."""
    return np.array(t).reshape(4, 4)

def generate_all_rank2_matrices():
    """Generate all rank-2 4×4 matrices over F3 using {-1, 0, 1}."""
    rank2_matrices = {}
    
    print("Generating all rank-2 matrices over F3...")
    count = 0
    
    # Try all possible pairs of rows as basis
    for basis_rows in combinations(range(4), 2):
        i1, i2 = basis_rows
        
        # Choose values for the two basis rows (using -1, 0, 1)
        for row1_vals in product([-1, 0, 1], repeat=4):
            row1 = np.array(row1_vals)
            if np.all(row1 == 0):  # Skip zero row
                continue
                
            for row2_vals in product([-1, 0, 1], repeat=4):
                row2 = np.array(row2_vals)
                
                # Check linear independence
                if np.all(row2 == 0):  # Skip zero row
                    continue
                
                # Check if row2 is multiple of row1
                if np.any(row1 != 0):
                    # Find first non-zero element in row1
                    nz_idx = np.nonzero(row1)[0][0]
                    if row1[nz_idx] != 0:
                        potential_mult = row2[nz_idx] / row1[nz_idx]
                        if np.allclose(row2, potential_mult * row1):
                            continue  # row2 is multiple of row1
                
                # Generate other two rows as linear combinations
                for coeffs in product([-1, 0, 1], repeat=4):
                    a, b, c, d = coeffs
                    
                    # Build full matrix with rows in correct positions
                    M = np.zeros((4, 4), dtype=int)
                    M[i1] = row1
                    M[i2] = row2
                    
                    # Fill remaining rows
                    other_rows = [r for r in range(4) if r not in basis_rows]
                    M[other_rows[0]] = mod3(a * row1 + b * row2)
                    M[other_rows[1]] = mod3(c * row1 + d * row2)
                    
                    # Convert to {-1, 0, 1} and verify rank
                    M = mod3(M)
                    if rank_f3(M) == 2:
                        key = matrix_to_tuple(M)
                        rank2_matrices[key] = M
                        count += 1
                        
                        if count % 10000 == 0:
                            print(f"  Generated {count} unique rank-2 matrices...")
    
    print(f"Total unique rank-2 matrices: {len(rank2_matrices)}")
    return rank2_matrices

def hadamard_divide_f3(M, A):
    """
    Compute Hadamard division M / A in F3 using {-1, 0, 1}.
    Returns None if division would involve dividing by zero.
    """
    result = np.zeros_like(M)
    for i in range(4):
        for j in range(4):
            if A[i, j] == 0:
                if M[i, j] != 0:
                    return None  # Can't divide non-zero by zero
                result[i, j] = 0  # 0/0 - we'll set to 0 (any value works)
            else:
                # In F3 with {-1, 0, 1}: 1^(-1) = 1, (-1)^(-1) = -1
                # Simply: x / y = x * y in this representation!
                result[i, j] = M[i, j] * A[i, j]
    return result

def check_expressibility_f3(M, rank2_matrices: Dict[tuple, np.ndarray]):
    """Check if matrix M can be expressed as Hadamard product of rank-2 matrices over F3."""
    print(f"Checking matrix over F3:")
    print(M)
    
    m_rank = rank_f3(M)
    print(f"\nRank: {m_rank}")
    
    if m_rank <= 2:
        print("Matrix has rank ≤ 2, so it's trivially expressible.")
        return True, (M, np.ones((4, 4), dtype=int))
    
    m_tuple = matrix_to_tuple(M)
    
    # Check if M itself is in our rank-2 set (shouldn't be if rank > 2)
    if m_tuple in rank2_matrices:
        print("Matrix is rank-2 (this shouldn't happen if rank calculation is correct)")
        return True, (M, np.ones((4, 4), dtype=int))
    
    # Try all rank-2 matrices as potential factors
    print("\nSearching for factorization...")
    checked = 0
    
    for a_tuple, A in rank2_matrices.items():
        checked += 1
        if checked % 50000 == 0:
            print(f"  Checked {checked}/{len(rank2_matrices)} matrices...")
        
        # Try to compute B = M / A
        B = hadamard_divide_f3(M, A)
        
        if B is None:  # Division failed due to zero pattern
            continue
        
        # Check if B is rank-2
        b_tuple = matrix_to_tuple(B)
        if b_tuple in rank2_matrices:
            print(f"\nFound factorization!")
            print(f"A = \n{A}")
            print(f"B = \n{B}")
            
            # Verify
            product = (A * B) % 3
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
            print("python check_matrix_f3.py")
            print("Then enter a 4x4 matrix over F3, e.g.:")
            print("1 2 0 1")
            print("2 1 1 0")
            print("0 1 2 2")
            print("1 0 2 1")
            return
        elif sys.argv[1] == '--generate-only':
            # Just generate and save the rank-2 matrices
            rank2_matrices = generate_all_rank2_matrices()
            import pickle
            with open('rank2_matrices_f3.pkl', 'wb') as f:
                pickle.dump(rank2_matrices, f)
            print("Saved rank-2 matrices to rank2_matrices_f3.pkl")
            return
    
    # Try to load precomputed rank-2 matrices
    try:
        import pickle
        print("Loading precomputed rank-2 matrices...")
        with open('rank2_matrices_f3.pkl', 'rb') as f:
            rank2_matrices = pickle.load(f)
        print(f"Loaded {len(rank2_matrices)} rank-2 matrices")
    except FileNotFoundError:
        print("No precomputed matrices found. Generating...")
        start_time = time.time()
        rank2_matrices = generate_all_rank2_matrices()
        print(f"Generation took {time.time() - start_time:.2f} seconds")
        
        # Save for future use
        import pickle
        with open('rank2_matrices_f3.pkl', 'wb') as f:
            pickle.dump(rank2_matrices, f)
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
    is_expressible, factorization = check_expressibility_f3(M, rank2_matrices)
    print(f"\nSearch took {time.time() - start_time:.2f} seconds")
    
    if is_expressible:
        print("\nResult: This matrix is EXPRESSIBLE as A ∘ B with rank(A) = rank(B) = 2")
    else:
        print("\nResult: This matrix is a COUNTEREXAMPLE")
        print("It cannot be expressed as A ∘ B with rank(A), rank(B) ≤ 2 over F3")

if __name__ == "__main__":
    main()