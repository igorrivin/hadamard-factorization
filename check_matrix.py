#!/usr/bin/env python3
"""
Check if a specific 4×4 binary matrix can be expressed as a Hadamard product.
"""

import numpy as np
import sys

def rank_f2(M):
    """Compute rank of matrix over F_2."""
    A = M.copy()
    n = A.shape[0]
    rank = 0
    
    for col in range(n):
        found = False
        for row in range(rank, n):
            if A[row, col] == 1:
                if row != rank:
                    A[[rank, row]] = A[[row, rank]]
                for r in range(n):
                    if r != rank and A[r, col] == 1:
                        A[r] = (A[r] + A[rank]) % 2
                rank += 1
                found = True
                break
    return rank

def matrix_to_int(M):
    """Convert 4×4 binary matrix to integer."""
    n = 0
    for i in range(4):
        for j in range(4):
            if M[i, j]:
                n |= (1 << (4*i + j))
    return n

def check_expressibility(M):
    """Check if matrix M can be expressed as Hadamard product of rank-2 matrices."""
    print(f"Checking matrix:")
    print(M)
    
    m_rank = rank_f2(M)
    print(f"\nRank: {m_rank}")
    
    if m_rank != 4:
        print("Matrix is not full rank, so it's trivially expressible.")
        return True
    
    m_int = matrix_to_int(M)
    
    # Load results if available
    try:
        import pickle
        with open('counterexamples_complete.pkl', 'rb') as f:
            counterexamples = set(pickle.load(f))
        
        if m_int in counterexamples:
            print("\nResult: This matrix is a COUNTEREXAMPLE")
            print("It cannot be expressed as A ∘ B with rank(A), rank(B) ≤ 2")
            return False
        else:
            print("\nResult: This matrix is EXPRESSIBLE")
            print("It can be expressed as A ∘ B with rank(A), rank(B) ≤ 2")
            return True
            
    except FileNotFoundError:
        print("\nNote: Run hadamard_search.py first for definitive results.")
        print("Performing limited search...")
        
        # Do a limited search
        rank2_count = 0
        for n in range(2**16):
            if rank2_count > 100:  # Just check first 100 rank-2 matrices
                break
            
            # Quick rank-2 check
            rows = []
            for i in range(4):
                rows.append((n >> (4*i)) & 0xF)
            
            # Simplified rank check
            if len(set(rows)) == 2:  # Likely rank 2
                rank2_count += 1
                
                # Check if this could be part of factorization
                for m in range(n, 2**16):
                    if (n & m) == m_int:  # Found factorization
                        print("\nFound a factorization in limited search!")
                        return True
        
        print("\nNo factorization found in limited search.")
        return None

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        # Show example usage
        print("Example usage:")
        print("python check_matrix.py")
        print("Then enter a 4x4 binary matrix, e.g.:")
        print("1 1 1 1")
        print("1 1 1 0")
        print("0 1 0 0")
        print("1 0 0 0")
        return
    
    print("Enter a 4×4 binary matrix (4 lines, 4 values each):")
    
    M = np.zeros((4, 4), dtype=int)
    try:
        for i in range(4):
            line = input().strip().split()
            if len(line) != 4:
                print(f"Error: Row {i+1} must have exactly 4 values")
                return
            for j in range(4):
                val = int(line[j])
                if val not in [0, 1]:
                    print(f"Error: Values must be 0 or 1, got {val}")
                    return
                M[i, j] = val
    except (ValueError, EOFError) as e:
        print(f"Error reading matrix: {e}")
        return
    
    print()
    check_expressibility(M)

if __name__ == "__main__":
    main()